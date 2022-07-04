#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import copy
import random
import shutil
from multiprocessing import Process

import numpy as np
import torch
import os
import time

from experiments.cdr_experiment import CDRsExperiments
from annoy import AnnoyIndex
from dataset.warppers import StreamingDatasetWrapper, DataSetWrapper
from utils.constant_pool import *
from utils.metrics_tool import MetricProcess


class SCDRTrainer(CDRsExperiments):
    def __init__(self, model, dataset_name, config_path, configs, result_save_dir, device='cuda:0',
                 log_path="log_streaming.txt"):
        CDRsExperiments.__init__(self, model, dataset_name, configs, result_save_dir, config_path, True, device, log_path)
        self.streaming_dataset = None
        self.finetune_epochs = 50
        self.minimum_finetune_data_num = 400
        self.finetune_data_num = 0
        self.cur_time = 0
        self.infer_model = None
        self.first_train_data_num = 0

    def update_batch_size(self, data_num):
        # TODO: 如何设置batch size也是需要研究的
        self.batch_size = max(min(data_num, self.configs.method_params.batch_size), int(data_num / 10))
        self.streaming_dataset.batch_size = self.batch_size
        self.model.update_neg_num(int(self.batch_size))

    def first_train(self, dataset: StreamingDatasetWrapper, epochs, ckpt_path=None):
        self.preprocess(load_data=False)
        self.first_train_data_num = dataset.total_data.shape[0]
        self.streaming_dataset = dataset
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

        self.infer_model = self.model.copy_network()

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
        self.queue_set.eval_data_queue.put([epoch, k, embedding_data, (epoch == self.epoch_num), False])

    def build_metric_tool(self):
        eval_used_data = self.train_loader.dataset.get_all_data()
        eval_used_data = np.reshape(eval_used_data, (eval_used_data.shape[0], np.product(eval_used_data.shape[1:])))

        self.metric_tool = MetricProcess(self.queue_set, self.message_queue, self.dataset_name, eval_used_data,
                                         self.train_loader.dataset.targets, None, None, None,
                                         self.result_save_dir, norm=self.is_image, k=self.fixed_k)
        self.metric_tool.start()

    def infer_embeddings(self, data):
        self.infer_model.to(self.device)
        data = torch.tensor(data, dtype=torch.float).to(self.device)
        with torch.no_grad():
            self.infer_model.eval()
            data_embeddings = self.infer_model(data).cpu()
            self.infer_model.train()
        return data_embeddings

    def resume_train(self, resume_epoch):
        embeddings = super(SCDRTrainer, self).resume_train(resume_epoch)
        self.infer_model = self.model.copy_network()
        return embeddings
