#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import time

import torch
from dataset.warppers import DataSetWrapper
from model.scdr.dependencies.experiment import Experiment
from model.dr_models.CDRs.cdr import CDRModel
from utils.constant_pool import *


class CDRsExperiments(Experiment):
    def __init__(self, clr_model, dataset_name, configs, result_save_dir, config_path, shuffle=True, device='cuda',
                 log_path="logs.txt", multi=False):
        Experiment.__init__(self, clr_model, dataset_name, configs, result_save_dir, config_path, shuffle, device,
                            log_path, multi)
        self.model = clr_model
        self.similar_num = 1
        self.clr_dataset = None
        self.debiased_sample = False
        self.resume_epochs = 0
        self.model.to(self.device)
        self.steps = 0
        self.init_epoch = self.resume_epochs if self.resume_epochs > 0 else self.epoch_num
        self.mixture_optimize = configs.method_params.method == "CDR"
        self.warmup_epochs = 0
        self.separation_epochs = 0
        self.accelerate = False

        self.knn_indices = None
        self.knn_dists = None
        if self.mixture_optimize:
            self.warmup_epochs = int(self.epoch_num * configs.method_params.separation_begin_ratio)
            self.separation_epochs = int(self.epoch_num * configs.method_params.steady_begin_ratio)

    def build_dataset(self, *args):
        # 数据加载器
        knn_cache_path = ConfigInfo.NEIGHBORS_CACHE_DIR.format(self.dataset_name, self.n_neighbors)
        pairwise_cache_path = ConfigInfo.PAIRWISE_DISTANCE_DIR.format(self.dataset_name)

        clr_dataset = DataSetWrapper(self.similar_num, self.batch_size)

        init_epoch = self.init_epoch
        if self.mixture_optimize:
            init_epoch = self.warmup_epochs

        self.train_loader, self.n_samples = clr_dataset.get_data_loaders(
            init_epoch, self.dataset_name, ConfigInfo.DATASET_CACHE_DIR, self.n_neighbors, knn_cache_path,
            pairwise_cache_path, self.is_image, multi=self.multi)

        self.knn_indices = clr_dataset.knn_indices
        self.knn_dists = clr_dataset.knn_distances

        self.batch_num = clr_dataset.batch_num
        self.model.batch_num = self.batch_num

    def forward(self, x, x_sim):
        # clr_model返回的是一个列表，分别是 x'、x、mu和var
        return self.model.forward(x, x_sim)

    def acquire_latent_code(self, inputs):
        if self.multi:
            return self.model.module.acquire_latent_code(inputs)

        return self.model.acquire_latent_code(inputs)

    def train(self, launch_time_stamp=None, target_metric_val=-1):
        batch_print_inter, self.vis_inter, ckp_save_inter = self._train_begin(launch_time_stamp)
        embeddings = None
        net = self.model
        if self.multi:
            net = net.module
        net.batch_num = self.batch_num
        training_loss_history = []
        test_loss_history = []

        # 迭代训练
        for epoch in range(self.start_epoch, self.epoch_num):

            sta = time.time()
            # epoch准备工作
            train_iterator, training_loss = self._before_epoch(epoch)
            # epoch迭代
            for idx, data in enumerate(train_iterator):
                self.steps += 1

                train_data = self._step_prepare(data, epoch, train_iterator)
                # mini batch迭代
                loss = self._train_step(*train_data)
                training_loss.append(loss.detach().cpu().numpy())
            self.train_time_cost += time.time() - sta
            # epoch扫尾工作
            embeddings = self._after_epoch(ckp_save_inter, epoch + 1, training_loss, training_loss_history,
                                           self.vis_inter)

        if not self.multi or self.device_id == 0:
            self._train_end(test_loss_history, training_loss_history, embeddings)
        return embeddings

    def resume_train(self, resume_epoch, *args):
        self.start_epoch = self.epoch_num
        self.epoch_num = self.start_epoch + resume_epoch
        if isinstance(self.model, CDRModel):
            self.model.update_separate_period(resume_epoch)
        self.tmp_log_file = open(self.tmp_log_path, "a")

        self.optimizer.param_groups[0]['lr'] = self.lr
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader),
                                                                    eta_min=0.00001, last_epoch=-1)
        return self.train()

    def _before_epoch(self, epoch):
        self.model = self.model.to(self.device)
        training_loss = []

        if self.mixture_optimize:
            if epoch == self.warmup_epochs:
                self.train_loader.dataset.transform.build_neighbor_repo(self.separation_epochs - self.warmup_epochs,
                                                                        self.n_neighbors)
            elif epoch == self.separation_epochs:
                self.train_loader.dataset.transform.build_neighbor_repo(self.epoch_num - self.separation_epochs,
                                                                        self.n_neighbors)

        train_iterator = iter(self.train_loader)
        if self.multi:
            self.train_loader.sampler.set_epoch(epoch)
        return train_iterator, training_loss

    def _after_epoch(self, ckp_save_inter, epoch, training_loss, training_loss_history, val_inter):
        ret_val = super()._after_epoch(ckp_save_inter, epoch, training_loss, training_loss_history, val_inter)
        return ret_val

    def _step_prepare(self, *args):
        data, epoch, train_iterator = args
        x, x_sim, indices, sim_indices = data[0]

        x = x.to(self.device, non_blocking=True)
        x_sim = x_sim.to(self.device, non_blocking=True)
        return x, x_sim, epoch, indices, sim_indices

    def _train_step(self, *args):
        x, x_sim, epoch, indices, sim_indices = args

        self.optimizer.zero_grad()

        with torch.cuda.device(self.device_id):
            _, x_embeddings, _, x_sim_embeddings = self.forward(x, x_sim)

        # 使用嵌入编码计算contrastive loss
        x_and_x_sim = torch.cat([x, x_sim], dim=0)

        net = self.model
        if self.multi:
            net = net.module
        train_loss = net.compute_loss(x_embeddings, x_sim_embeddings, epoch, x_and_x_sim, indices, sim_indices,
                                      self.train_loader.dataset.targets, self.result_save_dir, self.steps)

        train_loss.backward()
        # if self.steps % 5 == 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
        self.optimizer.step()
        return train_loss

    def model_prepare(self):
        self.model.preprocess()


