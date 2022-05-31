import os.path
import random
import time

import numpy as np
import torch
from sklearn.neighbors import LocalOutlierFactor
from annoy import AnnoyIndex

from dataset.warppers import StreamingDatasetWrapper
from experiments.scdr_trainer import SCDRTrainer
from utils.logger import InfoLogger
from utils.nn_utils import StreamingKNNSearcher, ANNOY, KD_TREE
from utils.scdr_utils import KeyPointsGenerater


def statistical_info(labels, previous_cls, pred_shifted_indices):
    gt_shifted_indices = []
    for i in range(len(labels)):
        if labels[i] not in previous_cls:
            gt_shifted_indices.append(i)

    detected_indices = np.intersect1d(gt_shifted_indices, pred_shifted_indices)
    pred_shifted_num = len(pred_shifted_indices)
    gt_shifted_num = len(gt_shifted_indices)
    detected_shifted_num = len(detected_indices)
    wrong_pred_num = pred_shifted_num - detected_shifted_num
    detect_ratio = detected_shifted_num / gt_shifted_num if gt_shifted_num > 0 else 0
    wrong_ratio = wrong_pred_num / pred_shifted_num
    output = "样本总数: %d 预测漂移数: %d 真实漂移数: %d 检出数: %d 误检数：%d 检出率: %.4f 误检率：%.4f" % (
        len(labels), pred_shifted_num, gt_shifted_num, detected_shifted_num, wrong_pred_num, detect_ratio, wrong_ratio)
    InfoLogger.info(output)

    wrong_indices = np.setdiff1d(pred_shifted_indices, gt_shifted_indices)
    wrong_cls, wrong_counts = np.unique(labels[wrong_indices], return_counts=True)
    tmp_indices = np.argsort(wrong_counts)[::-1]
    for i in tmp_indices:
        print("类别: {} 误检数 = {}".format(int(wrong_cls[i]), wrong_counts[i]))


class SCDRBase:
    def __init__(self, n_neighbors, model_trainer: SCDRTrainer, initial_train_num, initial_train_epoch, finetune_epoch,
                 finetune_data_rate=1.0, ckpt_path=None):
        self.n_neighbors = n_neighbors
        self.model_trainer = model_trainer
        self.ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
        self.initial_train_num = initial_train_num
        self.initial_train_epoch = initial_train_epoch
        self.finetune_epoch = finetune_epoch
        self.finetune_data_rate = finetune_data_rate
        self.minimum_finetune_data_num = 400

        self.pre_embeddings = None

        self.dataset = None
        self.cached_shift_indices = None
        self.key_data = None

        self.lof = None
        # self.knn_searcher = StreamingKNNSearcher(method=KD_TREE)
        self.knn_searcher = StreamingKNNSearcher(method=ANNOY)
        self.time_step = 0
        self.model_trained = False
        self.last_train_num = 0

        self.data_num_list = [0]

        self.shift_detect_time = 0
        self.knn_update_time = 0
        self.knn_cal_time = 0
        self.model_initial_time = 0
        self.model_repro_time = 0
        self.model_update_time = 0
        self.knn_approx_time = 0
        self.debug = True

    def fit_new_data(self, data, labels=None):
        pass

    def _initial_project(self, data, labels=None):
        self.dataset = StreamingDatasetWrapper(data, labels, self.model_trainer.batch_size)
        sta = time.time()
        self.pre_embeddings = self.model_trainer.first_train(self.dataset, self.initial_train_epoch, self.ckpt_path)
        self.model_initial_time = time.time() - sta
        self.model_trained = True
        self.last_train_num = self.pre_embeddings.shape[0]

    def _detect_distribution_shift(self, fit_data, pred_data, labels=None):
        """
        检测当前数据是否发生概念漂移
        :return:
        """
        sta = time.time()
        shifted_indices, data_embeddings = self._lof_based(fit_data, pred_data)
        self.shift_detect_time += time.time() - sta

        # if labels is not None:
        #     statistical_info(labels, np.unique(self.dataset.total_label[:self.last_train_num]), shifted_indices)

        cur_shift_indices = shifted_indices + self.dataset.pre_n_samples
        self.cached_shift_indices = cur_shift_indices if self.cached_shift_indices is None else \
            np.concatenate([self.cached_shift_indices, cur_shift_indices])

        return shifted_indices

    def _lof_based(self, fit_data, pred_data, n_neighbors=5, contamination=0.1):
        if self.lof is None:
            self.lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, metric="euclidean",
                                          contamination=contamination)

        self.lof.fit(fit_data)
        labels = self.lof.predict(pred_data)
        shifted_indices = np.where(labels == -1)[0]
        return shifted_indices, None

    def _process_shift_situation(self):
        self._update_training_data()

        sta = time.time()
        self.pre_embeddings = self.model_trainer.resume_train(self.finetune_epoch)
        self.model_update_time += time.time() - sta
        self.cached_shift_indices = None
        self.data_num_list = [0]
        return self.pre_embeddings

    def _update_training_data(self):
        # 更新knn graph
        sta = time.time()
        self.dataset.update_knn_graph(self.dataset.total_data[:self.last_train_num],
                                      self.dataset.total_data[self.last_train_num:], self.data_num_list)
        self.knn_update_time += time.time() - sta

        sampled_indices = self._sample_training_data()

        self.last_train_num = self.dataset.pre_n_samples
        self.model_trainer.update_batch_size(len(sampled_indices))
        # 更新neighbor_sample_repo以及训练集
        self.model_trainer.update_dataloader(self.finetune_epoch, sampled_indices)

    def _sample_training_data(self):
        # 采样训练子集
        n_samples = self.dataset.pre_n_samples
        # TODO: 可以选择随机采样或者基于聚类进行采样
        sampled_num = max(int(n_samples * self.finetune_data_rate), self.minimum_finetune_data_num)
        all_indices = np.arange(0, n_samples, 1)
        np.random.shuffle(all_indices)
        # TODO:可以选择与代表点数据进行拼接
        sampled_indices = all_indices[:sampled_num]
        if self.cached_shift_indices is not None:
            sampled_indices = np.union1d(sampled_indices, self.cached_shift_indices)
        return sampled_indices

    def _gather_data_stream(self):
        pass

    def save_model(self):
        self.model_trainer.save_weights(self.model_trainer.epoch_num)

    def print_time_cost_info(self):
        output_temp = 'Model Initialize: %.4f Model Update: %.4f Model Re-infer: %.4f \n' \
                      'kNN Calculate: %.4f kNN Update: %.4f kNN Approximate: %.4f \n' \
                      'Shift Detect: %.4f' % (self.model_initial_time, self.model_update_time,
                                              self.model_repro_time, self.knn_cal_time, self.knn_update_time,
                                              self.knn_approx_time, self.shift_detect_time)
        print(output_temp)


class SCDRModel(SCDRBase):
    def __init__(self, n_neighbors, buffer_size, model_trainer: SCDRTrainer, initial_train_num, initial_train_epoch,
                 finetune_epoch, finetune_data_rate=1.0, ckpt_path=None):
        SCDRBase.__init__(self, n_neighbors, model_trainer, initial_train_num, initial_train_epoch, finetune_epoch,
                          finetune_data_rate, ckpt_path)

        self.buffer_size = buffer_size
        self.initial_buffer_size = buffer_size
        # self.initial_buffer_size = 500

        self.buffered_data = None
        self.buffered_labels = None

    def _buffering(self, data, labels=None):
        buffer_size = self.buffer_size if self.model_trained else self.initial_buffer_size
        if self.buffered_data is None:
            self.buffered_data = data
            self.buffered_labels = labels
        else:
            self.buffered_data = np.concatenate([self.buffered_data, data], axis=0)
            if labels is not None:
                self.buffered_labels = np.concatenate([self.buffered_labels, labels])

        self.time_step += 1
        return self.buffered_data.shape[0] >= buffer_size

    def buffer_empty(self):
        return self.buffered_data is None

    def fit_new_data(self, data, labels=None):
        if not self._buffering(data, labels):
            return None

        return self.fit()

    def fit(self):
        buffer_data = self.buffered_data
        buffer_labels = self.buffered_labels
        if not self.model_trained:
            if buffer_data.shape[0] < self.initial_train_num:
                return None
            self._initial_project(buffer_data, buffer_labels)
            sta = time.time()
            self.knn_searcher.search(buffer_data, self.n_neighbors, query=False)
            self.knn_cal_time += time.time() - sta
        else:
            shifted_indices = self._detect_distribution_shift(self.dataset.total_data, buffer_data, buffer_labels)

            # 更新knn graph
            sta = time.time()
            nn_indices, nn_dists = self.knn_searcher.search(buffer_data, self.n_neighbors)
            self.knn_cal_time += time.time() - sta
            self.dataset.add_new_data(buffer_data, nn_indices, nn_dists, buffer_labels)
            self.data_num_list.append(buffer_data.shape[0] + self.data_num_list[-1])

            # if len(shifted_indices) <= int(buffer_data.shape[0] * 0.15):
            if len(self.cached_shift_indices) <= 30:
                # 此时应该将这些数据缓存起来
                sta = time.time()
                data_embeddings = self.model_trainer.infer_embeddings(buffer_data)
                self.model_repro_time += time.time() - sta
                self.pre_embeddings = np.concatenate([self.pre_embeddings, data_embeddings], axis=0)
            else:
                self._process_shift_situation()
        self.clear_buffer()

        return self.pre_embeddings

    def _gather_data_stream(self):
        # 1. 如果要进行采样的话，需要更新采样数据
        # 2. 更新knn graph
        # 3. 如果每次buffer后都需要评估的话，更新metric tool
        # 4. 更新一些缓存的数据，例如代表点
        pass

    def clear_buffer(self, final_embed=False):
        if self.buffered_data is not None and final_embed:
            # # 更新knn graph
            nn_indices, nn_dists = self.knn_searcher.search(self.buffered_data, self.n_neighbors)
            self.dataset.add_new_data(self.buffered_data, nn_indices, nn_dists, self.buffered_labels)

            self._process_shift_situation()
            self._gather_data_stream()
            return self.pre_embeddings

        self.buffered_data = None
        self.buffered_labels = None
