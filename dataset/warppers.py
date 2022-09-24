#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import math
import time

import numba.typed.typedlist
import numpy as np
from numba import jit
from scipy.spatial.distance import cdist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SubsetRandomSampler, Sampler

from dataset.samplers import SubsetDebiasedSampler, DebiasedSampler, CustomSampler
from dataset.transforms import CDRDataTransform
from dataset.datasets import *
from utils.constant_pool import ComponentInfo
from utils.logger import InfoLogger
from utils.math_utils import linear_search
from utils.nn_utils import compute_knn_graph
from utils.umap_utils import fuzzy_simplicial_set_partial


def build_dataset(data_file_path, dataset_name, is_image, normalize_method, root_dir):
    data_augment = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = None
    if is_image:
        if normalize_method == ComponentInfo.UMAP_NORMALIZE:
            train_dataset = UMAP_CLR_Image_Dataset(dataset_name, root_dir, True, data_augment, data_file_path)
            # test_dataset = UMAP_CLR_Image_Dataset(dataset_name, root_dir, False, data_augment, data_file_path)
        else:
            train_dataset = CLR_Image_Dataset(dataset_name, root_dir, True, data_augment, data_file_path)
            # test_dataset = CLR_Image_Dataset(dataset_name, root_dir, False, data_augment, data_file_path)
    else:
        if normalize_method == ComponentInfo.UMAP_NORMALIZE:
            train_dataset = UMAP_CLR_Text_Dataset(dataset_name, root_dir, True, data_file_path)
            # test_dataset = UMAP_CLR_Text_Dataset(dataset_name, root_dir, False, data_file_path)
        else:
            train_dataset = CLR_Text_Dataset(dataset_name, root_dir, True, data_file_path)
            # test_dataset = CLR_Text_Dataset(dataset_name, root_dir, False, data_file_path)
        data_augment = None
    return data_augment, test_dataset, train_dataset


class DataSetWrapper(object):
    batch_size: object

    def __init__(self, similar_num, batch_size):
        self.similar_num = similar_num
        self.batch_size = batch_size
        self.batch_num = 0
        self.test_batch_num = 0
        self.train_dataset = None
        self.knn_indices = None
        self.knn_distances = None
        self.symmetric_nn_indices = None
        self.symmetric_nn_weights = None
        self.raw_knn_weights = None
        self.n_neighbor = 0

    def get_data_loaders(self, epoch_num, dataset_name, root_dir, n_neighbors, knn_cache_path=None,
                         pairwise_cache_path=None, is_image=True, data_file_path=None, symmetric="TSNE", multi=False):
        self.n_neighbor = n_neighbors
        data_augment, test_dataset, train_dataset = build_dataset(data_file_path, dataset_name, is_image,
                                                                  ComponentInfo.UMAP_NORMALIZE, root_dir)
        self.train_dataset = train_dataset
        self.knn_indices, self.knn_distances = compute_knn_graph(train_dataset.data, knn_cache_path, n_neighbors,
                                                                 pairwise_cache_path, accelerate=False)

        self.distance2prob(ComponentInfo.UMAP_NORMALIZE, train_dataset, symmetric)

        train_num = self.update_transform(data_augment, epoch_num, is_image, train_dataset)

        train_indices = self._generate_train_indices(train_num, train_dataset)

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset, test_dataset,
                                                                            train_indices, [],
                                                                            False, multi)

        return train_loader, train_num

    def update_transform(self, data_augment, epoch_num, is_image, train_dataset):
        # 更新transform，迭代时对邻居进行采样返回正例对
        train_dataset.update_transform(CDRDataTransform(epoch_num, self.similar_num, train_dataset, is_image,
                                                        data_augment, self.n_neighbor, self.symmetric_nn_indices,
                                                        self.symmetric_nn_weights))
        train_num = train_dataset.data_num

        self.batch_num = math.floor(train_num / self.batch_size)
        return train_num

    def distance2prob(self, normalize_method, train_dataset, symmetric):
        # 针对高维空间中的点对距离进行处理，转换为0~1相似度并且进行对称化
        if normalize_method == ComponentInfo.UMAP_NORMALIZE:
            train_dataset.umap_process(self.knn_indices, self.knn_distances, self.n_neighbor, symmetric)
        else:
            train_dataset.simple_preprocess(self.knn_indices, self.knn_distances)

        self._update_knn_stat(train_dataset)

    def _update_knn_stat(self, train_dataset):
        self.symmetric_nn_indices = train_dataset.symmetry_knn_indices
        self.symmetric_nn_weights = train_dataset.symmetry_knn_weights
        # self.symmetric_nn_dists = train_dataset.symmetry_knn_dists
        # self.sym_no_norm_weights = train_dataset.sym_no_norm_weights
        self.raw_knn_weights = train_dataset.raw_knn_weights

    def get_train_validation_data_loaders(self, train_dataset, test_dataset, train_indices, val_indices,
                                          debiased_sample, multi):
        # obtain training indices that will be used for validation
        np.random.shuffle(train_indices)
        # np.random.shuffle(val_indices)
        InfoLogger.info("Train num = {} Val num = {}".format(len(train_indices), len(val_indices)))

        train_sampler, valid_sampler = self.construct_sampler(debiased_sample, test_dataset, train_dataset,
                                                              train_indices, val_indices, multi)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  drop_last=True, shuffle=False)

        # valid_loader = DataLoader(test_dataset, batch_size=self.batch_size, sampler=valid_sampler,
        #                           drop_last=True, shuffle=False)
        valid_loader = None
        return train_loader, valid_loader

    def construct_sampler(self, debiased_sample, test_dataset, train_dataset, train_indices, val_indices, multi):
        if debiased_sample:
            if multi:
                raise RuntimeError(
                    "distributed data parallel dose not support de-bias! Please set 'debiased_sample=False'")
            train_sampler = DebiasedSampler(train_dataset.processed_data, self.symmetric_nn_indices,
                                            self.batch_size)
            valid_sampler = DebiasedSampler(test_dataset.processed_data, self.symmetric_nn_indices,
                                            self.batch_size) if test_dataset is not None else None
        else:
            # define samplers for obtaining training and validation batches
            if multi:
                train_sampler = DistributedSampler(train_dataset)
                valid_sampler = DistributedSampler(test_dataset) if test_dataset is not None else None
            else:
                # train_sampler = SubsetRandomSampler(train_indices)
                # valid_sampler = SubsetRandomSampler(val_indices)
                train_sampler = CustomSampler(train_indices)
                valid_sampler = CustomSampler(val_indices) if test_dataset is not None else None
        return train_sampler, valid_sampler

    def _generate_train_indices(self, train_num, train_dataset):
        train_indices = list(range(train_num))
        return train_indices


avg_acc = []


def eval_knn_acc(acc_knn_indices, pre_knn_indices):
    acc_list = []
    n_neighbor = acc_knn_indices.shape[1]
    for i in range(acc_knn_indices.shape[0]):
        acc_list.append(len(np.intersect1d(acc_knn_indices[i], pre_knn_indices[i])) / n_neighbor)
        tmp = 0
        for j in range(n_neighbor):
            if pre_knn_indices[i][j] == acc_knn_indices[i][j]:
                tmp += 1
    # print("acc = %.4f" % np.mean(acc_list))
    global avg_acc
    avg_acc.append(np.mean(acc_list))
    print("average acc:", np.mean(avg_acc))
    return acc_list


class StreamingDatasetWrapper(DataSetWrapper):
    def __init__(self, initial_data, initial_label, batch_size, knn_indices=None, knn_dists=None):
        DataSetWrapper.__init__(self, 1, batch_size)
        # initial_data 是一个列表，第一个元素是数据，第二个元素是标签
        self.total_data = initial_data
        self.total_label = initial_label
        self.kNN_manager = KNNManager(self.n_neighbor)
        self.cur_n_samples = initial_data.shape[0] if initial_data is not None else 0
        self.cached_shifted_indices = []

    def _update_knn_stat(self, train_dataset):
        super()._update_knn_stat(train_dataset)

    def get_data_loaders(self, epoch_num, dataset_name, root_dir, n_neighbors, knn_cache_path=None,
                         pairwise_cache_path=None, is_image=True, data_file_path=None,
                         symmetric="TSNE", multi=False):
        self.n_neighbor = n_neighbors
        data_augment = transforms.Compose([
            transforms.ToTensor()
        ])

        data_augment, train_dataset = self.get_dataset(data_augment, is_image, ComponentInfo.UMAP_NORMALIZE)
        self.train_dataset = train_dataset
        if self.kNN_manager.is_empty():
            knn_indices, knn_distances = compute_knn_graph(train_dataset.data, None, n_neighbors,
                                                           None, accelerate=False)
            self.kNN_manager.add_new_kNN(knn_indices, knn_distances)

        self.distance2prob(ComponentInfo.UMAP_NORMALIZE, train_dataset, symmetric)

        train_num = self.update_transform(data_augment, epoch_num, is_image, train_dataset)
        train_indices = self._generate_train_indices(train_num, train_dataset)

        train_loader = self._get_train_data_loader(train_dataset, train_indices, multi)

        return train_loader, train_num

    def update_data_loaders(self, epoch_nums, sampled_indices, multi=False):

        self.train_dataset.transform.update(self.train_dataset, epoch_nums, self.symmetric_nn_indices,
                                            self.symmetric_nn_weights)

        train_loader = self._get_train_data_loader(self.train_dataset, sampled_indices, multi)
        return train_loader, len(sampled_indices)

    def get_dataset(self, data_augment, is_image, normalize_method):
        if is_image:
            if normalize_method == ComponentInfo.UMAP_NORMALIZE:
                train_dataset = UMAP_CLR_Image_Dataset(None, None, True, data_augment,
                                                       train_data=[self.total_data, self.total_label])
            else:
                train_dataset = CLR_Image_Dataset(None, None, True, data_augment,
                                                  train_data=[self.total_data, self.total_label])
        else:
            if normalize_method == ComponentInfo.UMAP_NORMALIZE:
                train_dataset = UMAP_CLR_Text_Dataset(None, None, True, train_data=[self.total_data, self.total_label])
            else:
                train_dataset = CLR_Text_Dataset(None, None, True, train_data=[self.total_data, self.total_label])
            data_augment = None
        return data_augment, train_dataset

    def _get_train_data_loader(self, train_dataset, train_indices, multi):
        InfoLogger.info("Train num = {}".format(len(train_indices)))
        np.random.shuffle(train_indices)
        train_sampler = CustomSampler(train_indices) if not multi else DistributedSampler(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  drop_last=True, shuffle=False)
        return train_loader

    def add_new_data(self, new_data, knn_indices=None, knn_dists=None, new_label=None):
        # 当数据积累到一定程度，需要更新模型的时候，才会为新的数据计算knn图
        self.cur_n_samples += new_data.shape[0]
        self.total_data = np.concatenate([self.total_data, new_data], axis=0)
        if new_label is not None:
            self.total_label = np.concatenate([self.total_label, new_label], axis=0)

        self.kNN_manager.add_new_kNN(knn_indices, knn_dists)
        self.train_dataset.add_new_data(new_data, new_label)

    def buffer_empty(self):
        return len(self.cached_shifted_indices) == 0

    def update_knn_graph(self, pre_data, new_data, data_num_list, cut_num=None):
        total_data = self.total_data if cut_num is None else self.total_data[:cut_num]
        knn_distances = self.kNN_manager.knn_dists if cut_num is None else self.kNN_manager.knn_dists[:cut_num]
        knn_indices = self.kNN_manager.knn_indices if cut_num is None else self.kNN_manager.knn_indices[:cut_num]

        new_n_samples = new_data.shape[0]
        # O(m*n*D)
        dists = cdist(new_data, total_data)

        pre_n_samples = pre_data.shape[0]
        neighbor_changed_indices = list(np.arange(0, new_n_samples, 1) + pre_n_samples)

        # acc_knn_indices, acc_knn_dists = compute_knn_graph(self.total_data, None, self.n_neighbor, None)
        # pre_acc_list, pre_a_acc_list = eval_knn_acc(acc_knn_indices, self.knn_indices, new_n_samples, pre_n_samples)

        neighbor_changed_indices = self.kNN_manager.update_previous_kNN(new_n_samples, pre_n_samples, dists,
                                                                        data_num_list, neighbor_changed_indices)

        self.raw_knn_weights = np.concatenate([self.raw_knn_weights, np.zeros((new_n_samples, self.n_neighbor))],
                                              axis=0)
        self.symmetric_nn_weights = np.concatenate([self.symmetric_nn_weights, np.empty(shape=new_n_samples)])
        self.symmetric_nn_indices = np.concatenate([self.symmetric_nn_indices, np.empty(shape=new_n_samples)])

        # sta = time.time()
        umap_graph, sigmas, rhos, self.raw_knn_weights = fuzzy_simplicial_set_partial(knn_indices, knn_distances,
                                                                                      self.raw_knn_weights,
                                                                                      neighbor_changed_indices)
        # print("fuzzy", time.time() - sta)

        # sta = time.time()
        updated_sym_nn_indices, updated_symm_nn_weights = extract_csr(umap_graph, neighbor_changed_indices)
        # print("extract", time.time() - sta)

        self.symmetric_nn_weights[neighbor_changed_indices] = updated_symm_nn_weights
        self.symmetric_nn_indices[neighbor_changed_indices] = updated_sym_nn_indices


# 负责存储以及更新kNN
class KNNManager:
    def __init__(self, k):
        self.k = k
        self.knn_indices = None
        self.knn_dists = None

    def is_empty(self):
        return self.knn_indices is None

    def add_new_kNN(self, new_knn_indices, new_knn_dists):
        if self.knn_indices is None:
            self.knn_indices = new_knn_indices
            self.knn_dists = new_knn_dists
            return

        if new_knn_indices is not None:
            self.knn_indices = np.concatenate([self.knn_indices, new_knn_indices], axis=0)
        if new_knn_dists is not None:
            self.knn_dists = np.concatenate([self.knn_dists, new_knn_dists], axis=0)

    def update_previous_kNN(self, new_data_num, pre_n_samples, dists2pre_data, data_num_list=None,
                            neighbor_changed_indices=None, symm=True):
        farest_neighbor_dist = self.knn_dists[:, -1]

        if neighbor_changed_indices is None:
            neighbor_changed_indices = []

        tmp_index = 0
        for i in range(new_data_num):
            indices = np.where(dists2pre_data[i] < farest_neighbor_dist)[0]

            if data_num_list is not None:
                # kNN计算正确的情况下，新数据i只可能改变在它之前到达的数据的kNN
                if i > 0 and i == data_num_list[tmp_index + 1]:
                    tmp_index += 1
                indices = indices[np.where(indices < data_num_list[tmp_index] + pre_n_samples)[0]]

                if len(indices) < 1:
                    continue

            for j in indices:
                if j not in neighbor_changed_indices:
                    neighbor_changed_indices.append(j)
                # 为当前元素找到一个插入位置即可，即distances中第一个小于等于dists[i][j]的元素位置，始终保持distances有序，那么最大的也就是最后一个
                insert_index = self.knn_dists.shape[1] - 1
                while insert_index >= 0 and dists2pre_data[i][j] <= self.knn_dists[j][insert_index]:
                    insert_index -= 1

                if symm and self.knn_indices[j][-1] not in neighbor_changed_indices:
                    neighbor_changed_indices.append(self.knn_indices[j][-1])

                # 这个更新的过程应该是迭代的，distance必须是递增的, 将[insert_index+1: -1]的元素向后移一位
                arr_move_one(self.knn_dists[j], insert_index + 1, dists2pre_data[i][j])
                arr_move_one(self.knn_indices[j], insert_index + 1, pre_n_samples + i)

        return neighbor_changed_indices


def extract_csr(csr_graph, indices, norm=True):
    nn_indices = []
    nn_weights = []

    for i in indices:
        pre = csr_graph.indptr[i]
        idx = csr_graph.indptr[i + 1]
        cur_indices = csr_graph.indices[pre:idx]
        cur_weights = csr_graph.data[pre:idx]

        nn_indices.append(cur_indices)
        if norm:
            nn_weights.append(cur_weights / np.sum(cur_weights))
        else:
            nn_weights.append(cur_weights)
    return nn_indices, nn_weights


# 将index之后的元素向后移动一位
def arr_move_one(arr, index, index_val):
    arr[index+1:] = arr[index:-1]
    arr[index] = index_val
