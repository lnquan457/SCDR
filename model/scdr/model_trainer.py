#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import shutil
from multiprocessing import Process

import torch
import os
import time

from model.scdr.dependencies.cdr_experiment import CDRsExperiments
from dataset.warppers import StreamingDatasetWrapper
from utils.constant_pool import *
from utils.metrics_tool import MetricProcess
from utils.queue_set import ModelUpdateQueueSet
from model.scdr.dependencies.scdr_utils import RepDataSampler


class SCDRTrainer(CDRsExperiments):
    def __init__(self, model, dataset_name, config_path, configs, result_save_dir, device='cuda:0',
                 log_path="log_streaming.txt"):
        CDRsExperiments.__init__(self, model, dataset_name, configs, result_save_dir, config_path, True, device,
                                 log_path)
        self.streaming_dataset = None
        self.finetune_epochs = 50
        self.minimum_finetune_data_num = 400
        self.finetune_data_num = 0
        self.cur_time = 0
        self.first_train_data_num = 0

    def update_batch_size(self, data_num):
        # TODO: 如何设置batch size也是需要研究的
        self.batch_size = max(min(data_num, self.configs.method_params.batch_size), int(data_num / 5))
        self.streaming_dataset.batch_size = self.batch_size
        self.model.update_batch_size(int(self.batch_size))

    def initialize_streaming_dataset(self, dataset):
        self.first_train_data_num = dataset.total_data.shape[0]
        self.streaming_dataset = dataset

    def first_train(self, dataset: StreamingDatasetWrapper, epochs, ckpt_path=None):
        self.preprocess(load_data=False)
        self.initialize_streaming_dataset(dataset)
        self.update_batch_size(self.first_train_data_num)
        self.update_dataloader(epochs)
        self.result_save_dir_modified = True
        self.do_test = False
        self.do_vis = False
        self.save_model = False
        self.save_final_embeddings = False
        self.draw_loss = False
        self.print_time_info = False
        self.result_save_dir = os.path.join(self.result_save_dir, "initial")
        if ckpt_path is None:
            self.epoch_num = epochs
            launch_time_stamp = int(time.time())
            self.pre_embeddings = self.train(launch_time_stamp)
        else:
            self.load_checkpoint(ckpt_path)
            self.pre_embeddings = self.visualize(None, device=self.device)[0]
            self._train_begin(int(time.time()))

        if self.config_path is not None:
            if not os.path.exists(self.result_save_dir):
                os.makedirs(self.result_save_dir)
            shutil.copyfile(self.config_path, os.path.join(self.result_save_dir, "config.yaml"))
        return self.pre_embeddings

    def update_dataloader(self, epochs, sampled_indices=None):
        if self.train_loader is None:
            self.train_loader, self.n_samples = self.streaming_dataset.get_data_loaders(
                epochs, self.dataset_name, ConfigInfo.DATASET_CACHE_DIR, self.n_neighbors, is_image=self.is_image,
                multi=self.multi)
        else:
            if sampled_indices is None:
                sampled_indices = np.arange(0, self.streaming_dataset.total_data.shape[0], 1)
            self.train_loader, self.n_samples = self.streaming_dataset.update_data_loaders(epochs, sampled_indices)

    def quantitative_test_all(self, epoch, embedding_data=None, mid_embeddings=None, device='cuda', val=False):
        self.metric_tool = None
        embedding_data, k, = self.quantitative_test_preprocess(embedding_data, device)[:2]
        # 向评估进程传输评估数据
        self.model_update_queue_set.eval_data_queue.put([epoch, k, embedding_data, (epoch == self.epoch_num), False])

    def build_metric_tool(self):
        eval_used_data = self.train_loader.dataset.get_all_data()
        eval_used_data = np.reshape(eval_used_data, (eval_used_data.shape[0], np.product(eval_used_data.shape[1:])))

        self.metric_tool = MetricProcess(self.model_update_queue_set, self.message_queue, self.dataset_name, eval_used_data,
                                         self.train_loader.dataset.targets, None, None, None,
                                         self.result_save_dir, norm=self.is_image, k=self.fixed_k)
        self.metric_tool.start()


class SCDRTrainerProcess(Process, SCDRTrainer):
    def __init__(self, cal_time_queue_set, model_update_queue_set, model, dataset_name, config_path, configs, result_save_dir, device='cuda:0',
                 log_path="log_streaming.txt", finetune_data_rate=0.3, finetune_epoch=50):
        self.name = "SCDR模型更新进程"
        Process.__init__(self, name=self.name)
        SCDRTrainer.__init__(self, model, dataset_name, config_path, configs, result_save_dir, device, log_path)
        self.cal_time_queue_set = cal_time_queue_set
        self.model_update_queue_set = model_update_queue_set
        self.data_sampler = RepDataSampler(self.n_neighbors, finetune_data_rate, minimum_sample_data_num=400,
                                           time_based_sample=False, metric_based_sample=False)
        self.update_count = 0
        self.finetune_epoch = finetune_epoch
        self.data_stream_ended = False

    def initialize_streaming_dataset(self, dataset):
        self.first_train_data_num = self.streaming_dataset.total_data.shape[0]

    def run(self) -> None:
        while True:

            # 因为这里是if，所以该进程不会在这里阻塞，
            if not self.model_update_queue_set.flag_queue.empty():
                flag = self.model_update_queue_set.flag_queue.get()
                if flag == ModelUpdateQueueSet.SAVE:
                    self.save_weights(self.epoch_num)
                elif flag == ModelUpdateQueueSet.DATA_STREAM_END:
                    print("data stream end!")
                    self.data_stream_ended = True

            if not self.data_stream_ended:
                # 模型更新的时候，这里是阻塞的，会积累很多数据，需要全部拿出来
                raw_info = self.model_update_queue_set.raw_data_queue.get()

                self.data_sampler.update_sample_weight(raw_info[0].shape[0])

                if self.streaming_dataset is None:
                    new_data, new_labels = raw_info
                    self.streaming_dataset = StreamingDatasetWrapper(new_data, new_labels, self.batch_size)
                else:
                    new_data, nn_indices, nn_dists, new_labels = raw_info
                    self.streaming_dataset.add_new_data(new_data, nn_indices, nn_dists, new_label=new_labels)
                # print("get data", new_data.shape)
                if self.model_update_queue_set.training_data_queue.empty():
                    continue

            # 该进程会在这里阻塞住
            training_info = self.model_update_queue_set.training_data_queue.get()
            # print("准备更新模型！")
            if self.update_count == 0:
                sta = time.time()
                embeddings = self.first_train(*training_info)
                self.cal_time_queue_set.model_initial_queue.put(time.time() - sta)
                ret = [embeddings, self.model.copy_network().cpu()]
            else:
                fitted_data_num, data_num_list, cur_data_num, pre_embeddings, must_indices = training_info
                self._get_all_from_raw_data_queue(cur_data_num - self.streaming_dataset.total_data.shape[0])
                pre_n_samples = pre_embeddings.shape[0]

                sta = time.time()
                self.streaming_dataset.update_knn_graph(self.streaming_dataset.total_data[:fitted_data_num],
                                                        self.streaming_dataset.total_data[fitted_data_num:],
                                                        data_num_list, cur_data_num)
                # print("new data num:", cur_data_num - fitted_data_num, " cal time:", time.time() - sta)
                self.cal_time_queue_set.knn_update_queue.put(time.time() - sta)

                sta = time.time()
                sampled_indices = \
                    self.data_sampler.sample_training_data(self.streaming_dataset.total_data[:pre_n_samples],
                                                           pre_embeddings,
                                                           self.streaming_dataset.knn_indices[:pre_n_samples],
                                                           must_indices, cur_data_num)
                self.cal_time_queue_set.training_data_sample_queue.put(time.time() - sta)

                self.update_batch_size(len(sampled_indices))

                self.update_dataloader(self.finetune_epoch, sampled_indices)

                sta = time.time()
                self.resume_train(self.finetune_epoch)
                ret = [cur_data_num, self.model.copy_network().cpu()]
                self.cal_time_queue_set.model_update_queue.put(time.time() - sta)

            self.model_update_queue_set.embedding_queue.put(ret)
            self.model_update_queue_set.MODEL_UPDATING.value = False
            self.update_count += 1

    def _get_all_from_raw_data_queue(self, target_num):
        # print("target num", target_num, " current num", self.streaming_dataset.total_data.shape[0])
        if target_num <= 0:
            return

        num = 0
        while num < target_num:
            new_data, nn_indices, nn_dists, new_labels = self.model_update_queue_set.raw_data_queue.get()
            self.streaming_dataset.add_new_data(new_data, nn_indices, nn_dists, new_label=new_labels)
            num += new_data.shape[0]


class IncrementalCDREx(SCDRTrainer):
    def __init__(self, clr_model, dataset_name, configs, result_save_dir, config_path, shuffle=True, device='cuda',
                 log_path="logs.txt", multi=False):
        CDRsExperiments.__init__(self, clr_model, dataset_name, configs, result_save_dir, config_path, shuffle, device,
                                 log_path, multi)
        self._rep_old_data = None
        self._pre_rep_old_embeddings = None
        self._rep_embeddings_pw_dists = []
        self._is_incremental_learning = False
        self.stream_dataset = None
        self.incremental_steps = 0

        # 用于计算VC损失
        self.rep_batch_nums = None
        self.rep_cluster_indices = None
        self.rep_exclude_indices = None

    def active_incremental_learning(self):
        self._is_incremental_learning = True

    def update_neg_num(self, neg_num=None):
        neg_num = self.model.neg_num if neg_num is None else neg_num
        self.model.update_neg_num(neg_num)

    def update_train_loader(self, train_indices):
        self.train_loader, self.n_samples = \
            self.clr_dataset.get_train_validation_data_loaders(self.clr_dataset.train_dataset, None,
                                                               train_indices, [], False, False)

    def resume_train(self, resume_epoch, *args):
        rep_args = args[0]
        self._update_rep_data_info(*rep_args)

        self.pre_embeddings = super().resume_train(resume_epoch)
        return self.pre_embeddings

    def _train_step(self, *args):
        x, x_sim, epoch, indices, sim_indices = args
        idx = self.incremental_steps % self.rep_batch_nums if self.rep_batch_nums is not None else -1
        self.optimizer.zero_grad()

        with torch.cuda.device(self.device_id):
            _, x_embeddings, _, x_sim_embeddings = self.forward(x, x_sim)
            rep_old_embeddings = self.model.acquire_latent_code(self._rep_old_data[idx] if idx >= 0 else None)

        # 使用嵌入编码计算contrastive loss
        if self._is_incremental_learning:
            train_loss = self.model.compute_loss(x_embeddings, x_sim_embeddings, epoch, self._is_incremental_learning,
                                                 rep_old_embeddings, self._pre_rep_old_embeddings[idx],
                                                 self.rep_cluster_indices[idx], self.rep_exclude_indices[idx],
                                                 self._rep_embeddings_pw_dists[idx])
        else:
            train_loss = self.model.compute_loss(x_embeddings, x_sim_embeddings, epoch, self._is_incremental_learning)

        train_loss.backward()
        self.incremental_steps += 1
        self.optimizer.step()
        return train_loss

    def build_dataset(self, *args):
        new_data, new_labels = args[0], args[1]
        self.stream_dataset = StreamingDatasetWrapper(new_data, new_labels, self.batch_size, self.n_neighbors)

    def _update_rep_data_info(self, rep_batch_nums, data, embeddings, cluster_indices, exclude_indices):
        self.incremental_steps = 0
        self._rep_embeddings_pw_dists = []
        self.rep_batch_nums = rep_batch_nums
        self._rep_old_data = torch.tensor(data, dtype=torch.float).to(self.device)
        if not isinstance(embeddings, torch.Tensor):
            self._pre_rep_old_embeddings = torch.tensor(embeddings, dtype=torch.float).to(self.device)
        for item in self._pre_rep_old_embeddings:
            self._rep_embeddings_pw_dists.append(torch.cdist(item, item))
        self.rep_cluster_indices = cluster_indices
        self.rep_exclude_indices = exclude_indices
