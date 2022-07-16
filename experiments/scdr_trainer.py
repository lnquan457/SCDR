#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import copy
import random
import shutil
from multiprocessing import Process
from multiprocessing.managers import BaseManager

import numpy as np
import torch
import os
import time

from experiments.cdr_experiment import CDRsExperiments
from annoy import AnnoyIndex
from dataset.warppers import StreamingDatasetWrapper, DataSetWrapper
from utils.constant_pool import *
from utils.metrics_tool import MetricProcess
from utils.queue_set import ModelUpdateQueueSet
from utils.scdr_utils import DataSampler


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
        self.infer_model = None
        self.first_train_data_num = 0

    def update_batch_size(self, data_num):
        # TODO: 如何设置batch size也是需要研究的
        self.batch_size = max(min(data_num, self.configs.method_params.batch_size), int(data_num / 10))
        self.streaming_dataset.batch_size = self.batch_size
        self.model.update_neg_num(int(self.batch_size))

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


class SCDRTrainerProcess(Process, SCDRTrainer):
    def __init__(self, queue_set, model, dataset_name, config_path, configs, result_save_dir, device='cuda:0',
                 log_path="log_streaming.txt", finetune_data_rate=1.0):
        self.name = "SCDR模型更新进程"
        Process.__init__(self, name=self.name)
        SCDRTrainer.__init__(self, model, dataset_name, config_path, configs, result_save_dir, device, log_path)
        self.queue_set = queue_set
        self.data_sampler = DataSampler(self.n_neighbors, finetune_data_rate, minimum_sample_data_num=400,
                                        time_based_sample=False, metric_based_sample=False)
        self.update_count = 0
        self.data_stream_ended = False

    def initialize_streaming_dataset(self, dataset):
        self.first_train_data_num = self.streaming_dataset.total_data.shape[0]

    def run(self) -> None:
        while True:

            # 获取训练相关的数据，现在的问题在于tensor数据不能跨进程传输
            if not self.queue_set.flag_queue.empty():
                flag = self.queue_set.flag_queue.get()
                if flag == ModelUpdateQueueSet.SAVE:
                    self.save_weights(self.epoch_num)
                elif flag == ModelUpdateQueueSet.STOP:
                    print("scdr trainer exit!")
                    break
                elif flag == ModelUpdateQueueSet.DATA_STREAM_END:
                    print("data stream end!")
                    self.data_stream_ended = True

            if not self.data_stream_ended:
                # 模型更新的时候，这里是阻塞的，会积累很多数据，需要全部拿出来
                raw_info = self.queue_set.raw_data_queue.get()

                self.data_sampler.update_sample_weight(raw_info[0].shape[0])

                if self.streaming_dataset is None:
                    new_data, new_labels = raw_info
                    self.streaming_dataset = StreamingDatasetWrapper(new_data, new_labels, self.batch_size)
                else:
                    new_data, nn_indices, nn_dists, new_labels = raw_info
                    self.streaming_dataset.add_new_data(new_data, nn_indices, nn_dists, new_label=new_labels)
                # print("get data", new_data.shape)
                if self.queue_set.training_data_queue.empty():
                    continue

            training_info = self.queue_set.training_data_queue.get()
            print("准备更新模型！")
            if self.update_count == 0:
                self.queue_set.MODEL_UPDATING.value = True
                embeddings = self.first_train(*training_info)
                ret = [embeddings, self.infer_model.cpu()]
            else:
                finetune_epoch, fitted_data_num, data_num_list, cur_data_num, pre_embeddings, must_indices = training_info
                self._get_all_from_raw_data_queue(cur_data_num - self.streaming_dataset.total_data.shape[0])
                print("final num", self.streaming_dataset.total_data.shape[0])
                pre_n_samples = pre_embeddings.shape[0]
                self.streaming_dataset.update_knn_graph(self.streaming_dataset.total_data[:fitted_data_num],
                                                        self.streaming_dataset.total_data[fitted_data_num:],
                                                        data_num_list, cur_data_num)

                sampled_indices = \
                    self.data_sampler.sample_training_data(self.streaming_dataset.total_data[:pre_n_samples],
                                                           pre_embeddings,
                                                           self.streaming_dataset.knn_indices[:pre_n_samples],
                                                           must_indices, cur_data_num)

                self.update_batch_size(len(sampled_indices))

                self.update_dataloader(finetune_epoch, sampled_indices)

                sta = time.time()
                self.queue_set.MODEL_UPDATING.value = True
                self.resume_train(finetune_epoch)
                ret = [cur_data_num, self.infer_model.cpu()]
                # self.model_update_time += time.time() - sta

            self.queue_set.embedding_queue.put(ret)
            self.queue_set.MODEL_UPDATING.value = False
            self.update_count += 1

    def _get_all_from_raw_data_queue(self, target_num):
        print("target num", target_num, " current num", self.streaming_dataset.total_data.shape[0])
        if target_num <= 0:
            print("return")
            return

        num = 0
        while num < target_num:
            new_data, nn_indices, nn_dists, new_labels = self.queue_set.raw_data_queue.get()
            self.streaming_dataset.add_new_data(new_data, nn_indices, nn_dists, new_label=new_labels)
            num += new_data.shape[0]
