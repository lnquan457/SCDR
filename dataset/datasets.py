#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import random

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from torchvision import transforms as transforms

from utils.logger import InfoLogger
from utils.nn_utils import compute_knn_graph, compute_accurate_knn, cal_snn_similarity
from utils.umap_utils import fuzzy_simplicial_set, construct_edge_dataset, compute_local_membership

MACHINE_EPSILON = np.finfo(np.double).eps


class MyTextDataset(Dataset):
    def __init__(self, dataset_name, root_dir, train=True, data_file_path=None, train_data=None, test_data=None):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        if train and train_data is None:
            self.data_file_path = os.path.join(root_dir, dataset_name + ".h5") if data_file_path is None else data_file_path
        self.train = train
        self.data = None
        self.targets = None
        self.data_num = 0
        self.min_neighbor_num = 0
        self.symmetry_knn_indices = None
        self.symmetry_knn_weights = None
        self.symmetry_knn_dists = None
        self.transform = None
        self.__load_data(train_data, test_data)

    def __len__(self):
        return self.data.shape[0]

    def __load_data(self, train_data, test_data):
        if self.train and train_data is not None:
            self.data = train_data[0]
            self.targets = train_data[1]
            return
        elif not self.train and test_data is not None:
            self.data = test_data[0]
            self.targets = test_data[1]
            return

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # train_data, train_labels, test_data, test_labels = \
        #     load_local_h5_by_path(self.data_file_path, ['x', 'y', 'x_test', 'y_test'])

        if self.train:
            train_data, train_labels = \
                load_local_h5_by_path(self.data_file_path, ['x', 'y'])
            self.data = train_data
            self.targets = train_labels
        else:
            # test_data, test_labels = \
            #     load_local_h5_by_path(self.data_file_path, ['x_test', 'y_test'])
            self.data = test_data
            # self.targets = test_labels

        self.data_num = self.data.shape[0]

    def __getitem__(self, index):
        text, target = self.data[index], int(self.targets[index])
        text = torch.tensor(text, dtype=torch.float)

        return text, target

    def _check_exists(self):
        return os.path.exists(self.data_file_path)

    def update_transform(self, new_transform):
        self.transform = new_transform

    def get_data(self, index):
        res = self.data[index]
        return torch.tensor(res, dtype=torch.float)

    def get_label(self, index):
        return int(self.targets[index])

    def get_dims(self):
        return int(self.data.shape[1])

    def get_all_data(self, data_num=-1):
        if data_num == -1:
            return self.data
        else:
            return self.data[torch.randperm(self.data_num)[:data_num], :]

    def get_data_shape(self):
        return self.data[0].shape

    def simple_preprocess(self, knn_indices, knn_distances):
        min_dis = np.expand_dims(np.min(knn_distances, axis=1), 1).repeat(knn_indices.shape[1], 1)
        max_dis = np.expand_dims(np.max(knn_distances, axis=1), 1).repeat(knn_indices.shape[1], 1)
        knn_distances = (knn_distances - min_dis) / (max_dis - min_dis)
        normal_knn_weights = np.exp(-knn_distances ** 2)
        normal_knn_weights /= np.expand_dims(np.sum(normal_knn_weights, axis=1), 1). \
            repeat(knn_indices.shape[1], 1)

        n_samples, n_neighbors = knn_indices.shape
        csr_row = np.expand_dims(np.arange(0, n_samples, 1), 1).repeat(n_neighbors, 1).ravel()
        csr_nn_weights = csr_matrix((normal_knn_weights.ravel(), (csr_row, knn_indices[:, :n_neighbors].ravel())),
                                    shape=(n_samples, n_samples))
        symmetric_nn_weights = csr_nn_weights + csr_nn_weights.T
        nn_indices, nn_weights, self.min_neighbor_num, raw_weights, _ = get_kw_from_coo(symmetric_nn_weights,
                                                                                        n_neighbors,
                                                                                        n_samples)

        self.symmetry_knn_indices = np.array(nn_indices, dtype=object)
        self.symmetry_knn_weights = np.array(nn_weights, dtype=object)


class MyImageDataset(MyTextDataset):
    def __init__(self, dataset_name, root_dir, train=True, transform=None, data_file_path=None, train_data=None,
                 test_data=None):
        MyTextDataset.__init__(self, dataset_name, root_dir, train, data_file_path, train_data, test_data)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = np.squeeze(img)
        mode = 'RGB' if len(img.shape) == 3 else 'L'
        if mode == 'RGB':
            img = Image.fromarray(img, mode=mode)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_data(self, index):
        res = self.data[index]
        res = res.astype(np.uint8)
        # res = Image.fromarray(np.squeeze(res), mode='L')
        # return torch.tensor(res, dtype=torch.float)
        return res

    def get_all_data(self, data_num=-1):
        if data_num == -1:
            return np.transpose(self.data, (0, 3, 1, 2))
        else:
            return np.transpose(self.data[torch.randperm(self.data_num)[:data_num], :, :, :], (0, 3, 1, 2))


class UMAP_Text_Dataset(MyTextDataset):
    def __init__(self, dataset_name, root_dir, train=True, repeat=1, data_file_path=None, train_data=None,
                 test_data=None):
        MyTextDataset.__init__(self, dataset_name, root_dir, train, data_file_path, train_data, test_data)
        self.repeat = repeat
        self.edge_data = None
        self.edge_num = None
        self.edge_weight = None
        self.raw_knn_weights = None

    def build_fuzzy_simplicial_set(self, knn_cache_path, pairwise_cache_path, n_neighbors, metric="euclidean",
                                   max_candidates=60):
        # 获得原数据的KNN近邻图
        knn_indices, knn_distances = compute_knn_graph(self.data, knn_cache_path, n_neighbors, pairwise_cache_path,
                                                       metric, max_candidates)

        InfoLogger.info("开始生成模糊单纯形集...")
        """
        umap_graph就是模糊单纯性集的1-skeleton表示，是一个权重矩阵，实际上是CSR格式存储的
        sigmas是平滑正则化因子，
        rhos是最近邻距离
        """
        umap_graph, sigmas, rhos, self.raw_knn_weights = fuzzy_simplicial_set(
            X=self.data,
            n_neighbors=n_neighbors,
            knn_indices=knn_indices,
            knn_dists=knn_distances)
        return umap_graph, sigmas, rhos

    def umap_process(self, knn_cache_path, pairwise_cache_path, n_neighbors, embedding_epoch, metric="euclidean",
                     max_candidates=60):
        # 这里的umap_graph是一共csr格式的矩阵，表示了所有可能存在边的数据点对关系以及它们的权重
        umap_graph, sigmas, rhos = self.build_fuzzy_simplicial_set(knn_cache_path, pairwise_cache_path, n_neighbors,
                                                                   metric, max_candidates)
        self.edge_data, self.edge_num, self.edge_weight = construct_edge_dataset(
            self.data, umap_graph, embedding_epoch)

        return self.edge_data, self.edge_num

    def __getitem__(self, index):
        to_data, from_data = self.edge_data[0][index], self.edge_data[1][index]

        return torch.tensor(to_data, dtype=torch.float), torch.tensor(from_data, dtype=torch.float)

    def __len__(self):
        return self.edge_num


class UMAP_Image_Dataset(MyImageDataset, UMAP_Text_Dataset):
    def __init__(self, dataset_name, root_dir, train=True, transform=None, repeat=1, data_file_path=None,
                 train_data=None, test_data=None):
        MyImageDataset.__init__(self, dataset_name, root_dir, train, transform, data_file_path, train_data, test_data)
        UMAP_Text_Dataset.__init__(self, dataset_name, root_dir, train, repeat, data_file_path, train_data, test_data)
        self.transform = transform

    def __getitem__(self, index):
        to_data, from_data = self.edge_data[0][index], self.edge_data[1][index]

        # to_data = np.transpose(to_data, [2, 0, 1])
        # from_data = np.transpose(from_data, [2, 0, 1])
        #
        # mode = 'RGB' if len(to_data.shape) == 3 else 'L'
        # if mode == 'RGB':
        #     to_data = Image.fromarray(to_data, mode=mode)
        #     from_data = Image.fromarray(from_data, mode=mode)

        if self.transform is not None:
            to_data = self.transform(to_data)
            from_data = self.transform(from_data)

        return to_data, from_data


class CLR_Text_Dataset(MyTextDataset):
    def __init__(self, dataset_name, root_dir, train=True, data_file_path=None, train_data=None, test_data=None):
        MyTextDataset.__init__(self, dataset_name, root_dir, train, data_file_path, train_data, test_data)

    def __getitem__(self, index):
        text, target = self.data[index], int(self.targets[index])
        x, x_sim, idx, sim_idx = self.transform(text, index)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
            x_sim = torch.tensor(x_sim, dtype=torch.float)
        return [x, x_sim, idx, sim_idx], target

    def sample_data(self, indices):
        x = self.data[indices]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        return x

    def add_new_data(self, data, labels=None):
        self.data = data if self.data is None else np.concatenate([self.data, data], axis=0)
        if labels is not None:
            self.targets = labels if self.targets is None else np.append(self.targets, labels)


class CLR_Image_Dataset(MyImageDataset):
    def __init__(self, dataset_name, root_dir, train=True, transform=None, data_file_path=None, train_data=None,
                 test_data=None):
        MyImageDataset.__init__(self, dataset_name, root_dir, train, transform, data_file_path, train_data, test_data)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = np.squeeze(img)
        mode = 'RGB' if len(img.shape) == 3 else 'L'
        img = Image.fromarray(img, mode=mode)
        if self.transform is not None:
            img = self.transform(img, index)

        return img, target

    def sample_data(self, indices):
        num = len(indices)
        first_data = self.data[indices[0]]
        ret_data = torch.empty((num, first_data.shape[2], first_data.shape[0], first_data.shape[1]))
        count = 0
        transform = transforms.ToTensor()
        for index in indices:
            img = np.squeeze(self.data[index])
            mode = 'RGB' if len(img.shape) == 3 else 'L'
            img = Image.fromarray(img, mode=mode)
            img = transform(img)
            ret_data[count, :, :, :] = img.unsqueeze(0)
            count += 1
        return ret_data


class UMAP_CLR_Text_Dataset(CLR_Text_Dataset):
    def __init__(self, dataset_name, root_dir, train=True, data_file_path=None, train_data=None, test_data=None):
        CLR_Text_Dataset.__init__(self, dataset_name, root_dir, train, data_file_path, train_data, test_data)
        self.umap_graph = None
        # 没有经过归一化和对称话的
        self.raw_knn_weights = None
        # 经过对称化但是没有归一化的
        self.sym_no_norm_weights = None
        self.min_neighbor_num = None
        self.knn_dist = None
        self.knn_indices = None

    def build_fuzzy_simplicial_set(self, knn_indices, knn_distances, n_neighbors, symmetric):
        InfoLogger.info("开始生成模糊单纯形集...")
        """
        umap_graph就是模糊单纯性集的1-skeleton表示，是一个权重矩阵，实际上是CSR格式存储的
        sigmas是平滑正则化因子，
        rhos是最近邻距离
        """
        # print(knn_indices[5009], self.targets[knn_indices[5009].astype(np.int)])
        self.umap_graph, sigmas, rhos, self.raw_knn_weights, knn_dist = fuzzy_simplicial_set(
            X=self.data,
            n_neighbors=n_neighbors, knn_indices=knn_indices,
            knn_dists=knn_distances, return_dists=True, symmetric=symmetric, labels=self.targets)

        # knn_cache_path = ConfigInfo.NEIGHBORS_CACHE_DIR.format(self.dataset_name, 3*n_neighbors)
        # pairwise_cache_path = ConfigInfo.PAIRWISE_DISTANCE_DIR.format(self.dataset_name)
        # knn_3 = compute_accurate_knn(self.data, 3 * n_neighbors, knn_cache_path, pairwise_cache_path)[0]
        #
        # snn_sim_3 = cal_snn_similarity(knn_3)[1]
        # sorted_indices = np.argsort(snn_sim_3, axis=1)[:, ::-1]
        # rows = np.expand_dims(np.arange(0, self.data_num, 1), axis=1).repeat(n_neighbors * 3, axis=1)
        # knn_based_on_snn = knn_3[rows, sorted_indices]
        #
        # total_num = self.data_num * n_neighbors
        # right_knn_neighbor_num = 0
        # right_snn_neighbor_num = 0
        # for i in range(self.data_num):
        #     for j in range(n_neighbors):
        #         knn_neighbor_idx = int(knn_indices[i][j])
        #         snn_neighbor_idx = int(knn_based_on_snn[i][j])
        #         if self.targets[knn_neighbor_idx] == self.targets[int(i)]:
        #             right_knn_neighbor_num += 1
        #         if self.targets[snn_neighbor_idx] == self.targets[int(i)]:
        #             right_snn_neighbor_num += 1
        # print("Origin KNN Neighbor Hit:", right_knn_neighbor_num / total_num, " Origin SNN Neighbor Hit:",
        #       right_snn_neighbor_num / total_num)

        self.symmetry_knn_dists = knn_dist.tocoo()

    def umap_process(self, knn_indices, knn_distances, n_neighbors, symmetric):
        # 这里的umap_graph是一共csr格式的矩阵，表示了所有可能存在边的数据点对关系以及它们的权重
        self.build_fuzzy_simplicial_set(knn_indices, knn_distances, n_neighbors, symmetric)

        self.data_num = knn_indices.shape[0]
        n_samples = self.data_num

        nn_indices, nn_weights, self.min_neighbor_num, raw_weights, nn_dists \
            = get_kw_from_coo(self.umap_graph, n_neighbors, n_samples, self.symmetry_knn_dists)

        self.symmetry_knn_indices = np.array(nn_indices, dtype=object)
        self.symmetry_knn_weights = np.array(nn_weights, dtype=object)
        self.symmetry_knn_dists = np.array(nn_dists, dtype=object)
        self.sym_no_norm_weights = np.array(raw_weights, dtype=object)


class UMAP_CLR_Image_Dataset(CLR_Image_Dataset, UMAP_CLR_Text_Dataset):
    def __init__(self, dataset_name, root_dir, train=True, transform=None, data_file_path=None, train_data=None,
                 test_data=None):
        CLR_Image_Dataset.__init__(self, dataset_name, root_dir, train, transform, data_file_path, train_data,
                                   test_data)
        UMAP_CLR_Text_Dataset.__init__(self, dataset_name, root_dir, train, train_data, test_data)


def get_kw_from_coo(csr_graph, n_neighbors, n_samples, dist_csr=None):
    nn_indices = []
    # 经过归一化的
    nn_weights = []
    # 未归一化的
    raw_weights = []
    nn_dists = []

    tmp_min_neighbor_num = n_neighbors
    for i in range(1, n_samples + 1):
        pre = csr_graph.indptr[i - 1]
        idx = csr_graph.indptr[i]
        cur_indices = csr_graph.indices[pre:idx]
        if dist_csr is not None:
            nn_dists.append(dist_csr.data[pre:idx])
        tmp_min_neighbor_num = min(tmp_min_neighbor_num, idx - pre)
        cur_weights = csr_graph.data[pre:idx]

        # re_indices = np.argsort(cur_weights)[::-1]
        # # 限制邻居采样的范围
        # re_indices = re_indices[:neighbor_range]
        # cur_weights = cur_weights[re_indices]
        # cur_indices = cur_indices[re_indices]

        nn_indices.append(cur_indices)
        cur_sum = np.sum(cur_weights)
        nn_weights.append(cur_weights / cur_sum)
        raw_weights.append(cur_weights)
    return nn_indices, nn_weights, tmp_min_neighbor_num, raw_weights, nn_dists


def load_local_h5_by_path(dataset_path, keys):
    f = h5py.File(dataset_path, "r")
    res = []
    for key in keys:
        if key in f.keys():
            res.append(f[key][:])
        else:
            res.append(None)
        # InfoLogger.info(key + ": " + str(f[key][:].shape))
    f.close()
    return res
