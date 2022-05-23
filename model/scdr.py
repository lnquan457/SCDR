import random

import numpy as np
import torch
from sklearn.neighbors import LocalOutlierFactor
from annoy import AnnoyIndex

from dataset.warppers import StreamingDatasetWrapper
from experiments.scdr_trainer import SCDRTrainer
from utils.logger import InfoLogger


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


class SCDRModel:
    def __init__(self, n_neighbors, buffer_size, model_trainer: SCDRTrainer, initial_train_epoch,
                 finetune_epoch, finetune_data_ratio=1.0, ckpt_path=None):
        self.n_neighbors = n_neighbors
        self.model_trainer = model_trainer
        self.ckpt_path = ckpt_path
        self.initial_train_epoch = initial_train_epoch
        self.finetune_epoch = finetune_epoch
        self.finetune_data_ratio = finetune_data_ratio
        self.minimum_finetune_data_num = 400
        self.buffer_size = buffer_size
        self.initial_buffer_size = buffer_size
        # self.initial_buffer_size = 500

        self.pre_embeddings = None
        self.buffered_data = None
        self.buffered_labels = None
        self.dataset = None
        self.pre_shift_indices = None

        self.lof = None
        self.annoy_index = None
        self.time_step = 0
        self.model_trained = False
        self.last_train_num = 0

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
            self._initial_project(buffer_data, buffer_labels)
            self._build_annoy_index(buffer_data)
        else:
            shifted_indices, data_embeddings = self._detect_distribution_shift(buffer_data, buffer_labels)
            cur_shift_indices = shifted_indices + self.dataset.pre_n_samples
            self.pre_shift_indices = cur_shift_indices if self.pre_shift_indices is None else \
                np.concatenate([self.pre_shift_indices, cur_shift_indices])

            # 更新knn graph
            nn_indices, nn_dists = self._get_knn(buffer_data)
            self.dataset.add_new_data(buffer_data, nn_indices, nn_dists, buffer_labels)

            # if len(shifted_indices) <= int(buffer_data.shape[0] * 0.15):
            if len(self.pre_shift_indices) <= 30:
                # 此时应该将这些数据缓存起来
                self.pre_embeddings = np.concatenate([self.pre_embeddings, data_embeddings], axis=0)
            else:
                self._process_shift_situation()
        self._gather_data_stream()
        return self.pre_embeddings

    def _initial_project(self, data, labels=None):
        self.dataset = StreamingDatasetWrapper(data, labels, self.model_trainer.batch_size)
        self.pre_embeddings = self.model_trainer.first_train(self.dataset, self.initial_train_epoch, self.ckpt_path)
        self.model_trained = True
        self.last_train_num = self.pre_embeddings.shape[0]

    def _detect_distribution_shift(self, data, labels=None):
        """
        检测当前数据是否发生概念漂移
        :return:
        """

        # shifted_indices, data_embeddings = self._nn_embeddings_sim_based(data)
        shifted_indices, data_embeddings = self._lof_based(self.dataset.total_data, data)
        if data_embeddings is None:
            data_embeddings = self.model_trainer.infer_embeddings(data)

        # if labels is not None:
        #     statistical_info(labels, np.unique(self.dataset.total_label[:self.last_train_num]), shifted_indices)

        return shifted_indices, data_embeddings

    def _nn_embeddings_sim_based(self, data, threshold=0.4):
        self._build_annoy_index(data)
        high_indices = []
        data_num = data.shape[0]
        for i in range(data_num):
            indices, distances = self.annoy_index.get_nns_by_vector(data[i], self.n_neighbors, include_distances=True)
            # self.annoy_index.add_item(self.annoy_index.get_n_items(), data[i])
            high_indices.append(indices)

            # acc_dist = np.linalg.norm(self.cur_dataset[0] - data[i], axis=1)
            # acc_sort_indices = np.argsort(acc_dist)
            # acc_high_indices = acc_sort_indices[:self.n_neighbors]
            # acc_high_dist = acc_dist[acc_high_indices]
            # high_indices.append(acc_high_indices)
            # high_distances.append(acc_high_dist)

        high_indices = np.array(high_indices)

        knn_embeddings = self.pre_embeddings[high_indices.ravel()].reshape((data_num, self.n_neighbors, -1))

        # knn_data = torch.tensor(self.cur_dataset[0][high_indices.ravel()], dtype=torch.float).to(self.device)
        # with torch.no_grad():
        #     self.model.eval()
        #     _, knn_embeddings = self.model.encode(knn_data)
        #     self.model.train()
        # knn_embeddings = knn_embeddings.cpu().reshape((data_num, self.n_neighbors, -1))

        data_embeddings = self.model_trainer.infer_embeddings(data)

        similarities = self.model_trainer.model.similarity_func(torch.tensor(knn_embeddings),
                                                                data_embeddings.unsqueeze(1).
                                                                repeat(1, self.n_neighbors, 1))[0]
        mean_sim = torch.mean(similarities, dim=1)

        # Todo: 如何确定threshold
        shifted_indices = torch.where(mean_sim < threshold)[0].numpy()
        return shifted_indices, data_embeddings

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

        self.pre_embeddings = self.model_trainer.resume_train(self.finetune_epoch)
        self.pre_shift_indices = None
        return self.pre_embeddings

    def _gather_data_stream(self):
        # 1. 如果要进行采样的话，需要更新采样数据
        # 2. 更新knn graph
        # 3. 如果每次buffer后都需要评估的话，更新metric tool
        # 4. 更新一些缓存的数据，例如代表点
        # self._build_annoy_index(data)
        self.clear_buffer()
        pass

    def clear_buffer(self, final_embed=False):
        if self.buffered_data is not None and final_embed:

            # # 更新knn graph
            nn_indices, nn_dists = self._get_knn(self.buffered_data)
            self.dataset.add_new_data(self.buffered_data, nn_indices, nn_dists, self.buffered_labels)

            self._process_shift_situation()
            self._gather_data_stream()
            return self.pre_embeddings

            # return self.model_trainer.infer_embeddings(np.concatenate([self.dataset.total_data, self.buffered_data],
            #                                                           axis=0)).numpy()
        self.buffered_data = None
        self.buffered_labels = None

    def _update_training_data(self):

        # 更新knn graph
        # nn_indices = self.dataset.knn_indices[self.last_train_num:]
        # nn_dists = self.dataset.knn_distances[self.last_train_num:]

        self.dataset.update_knn_graph(self.dataset.total_data[:self.last_train_num],
                                      self.dataset.total_data[self.last_train_num:], self.buffer_size)

        # 采样训练子集
        n_samples = self.dataset.pre_n_samples
        # TODO: 可以选择随机采样或者基于聚类进行采样
        sampled_num = max(int(n_samples * self.finetune_data_ratio), self.minimum_finetune_data_num)
        all_indices = np.arange(0, n_samples, 1)
        np.random.shuffle(all_indices)
        # TODO:可以选择与代表点数据进行拼接
        sampled_indices = all_indices[:sampled_num]
        if self.pre_shift_indices is not None:
            sampled_indices = np.union1d(sampled_indices, self.pre_shift_indices)
        self.last_train_num = n_samples

        self.model_trainer.update_batch_size(n_samples)
        # 更新neighbor_sample_repo以及训练集
        self.model_trainer.update_dataloader(self.finetune_epoch, sampled_indices)

    def _build_annoy_index(self, data):
        n_samples, dim = data.shape
        if self.annoy_index is None:
            pre_nums = 0
            self.annoy_index = AnnoyIndex(dim, 'euclidean')
        else:
            pre_nums = self.annoy_index.get_n_items()
            self.annoy_index.unbuild()

        for i in range(n_samples):
            self.annoy_index.add_item(i + pre_nums, data[i])
        self.annoy_index.build(100)

    def _get_knn(self, data):
        data_num = data.shape[0]
        nn_indices = np.empty((data_num, self.n_neighbors), dtype=int)
        nn_dists = np.empty((data_num, self.n_neighbors), dtype=float)
        self._build_annoy_index(data)
        for i in range(data_num):
            cur_indices, cur_dists = self.annoy_index.get_nns_by_vector(data[i], self.n_neighbors + 1,
                                                                        include_distances=True)
            nn_indices[i], nn_dists[i] = cur_indices[1:], cur_dists[1:]

        return nn_indices, nn_dists

    def save_model(self):
        self.model_trainer.save_weights(self.model_trainer.epoch_num)