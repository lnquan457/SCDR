#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
from pynndescent import NNDescent

from utils.umap_utils import find_ab_params, convert_distance_to_probability
import networkx as nx
import bisect
from scipy.spatial.distance import pdist, squareform

MACHINE_EPSILON = np.finfo(np.double).eps


def get_random_m(data, m):
    data = np.array(data)
    n_samples = data.shape[0]
    left = m
    tmp_set = set()
    # 随机选择锚点数据
    while len(tmp_set) < m:
        cur_random_index = np.floor(np.random.rand(left) * n_samples).astype(np.int).tolist()
        tmp_set.update(cur_random_index)
        left = m - len(tmp_set) + 500
    landmark_index = list(tmp_set)[:m]
    random_m_data = data[landmark_index]
    return random_m_data, landmark_index


def cal_tensor_pairwise_dist(x, square=True):
    """
    计算tensor矩阵之间的欧式距离，(a-b)^2 = a^2 + b^2 - 2*a*b
    :param square:
    :param x: n*m的tensor矩阵
    :return:
    """

    sum_x = torch.sum(torch.square(x), dim=1)
    sum_x2 = torch.sum(torch.square(x.T), dim=0)
    sum_x3 = -2 * torch.mm(x, x.T)
    dist = sum_x + sum_x2 - sum_x3
    dist[dist < 0] = 0
    if not square:
        dist = torch.sqrt(dist)
    return dist


def normalize_matrix(x, axis):
    """
    normalize matrix with formulation norm_x = (x - x_min) / (x_max - x_min)
    :param x: input matrix, [N, M]
    :param axis: 0 indicate normalize by rows and 1 for columns
    :return: normalized data
    """
    # x_max = np.max(x, axis=axis)
    # x_min = np.min(x, axis=axis)
    # normalized_data = (x - x_min) / (x_max - x_min + 1e-12)

    n = x.shape[0]
    normalized_data = x / np.expand_dims(np.sum(x, axis=axis), axis=1).repeat(n, 1)
    return normalized_data


def _student_t_similarity(rep1, rep2, *args):
    pairwise_matrix = torch.norm(rep1 - rep2, dim=-1)
    # 用自由度为1的t分布进行转换
    similarity_matrix = 1 / (1 + pairwise_matrix ** 2)
    return similarity_matrix, pairwise_matrix


def _exp_similarity(rep1, rep2, *args):
    pairwise_matrix = torch.norm(rep1 - rep2, dim=-1)
    # 用exp(-|x-y|^2)将距离转换为概率
    similarity_matrix = torch.exp(-pairwise_matrix ** 2)
    return similarity_matrix, pairwise_matrix


def _cosine_similarity(rep1, rep2, *args):
    x = rep2[0]
    x = F.normalize(x, dim=1)
    similarity_matrix = torch.matmul(x, x.T).clamp(min=1e-7)
    pairwise_matrix = torch.norm(rep1 - rep2, dim=-1)
    return similarity_matrix, pairwise_matrix


a = None
b = None
pre_min_dist = -1


def _umap_similarity(rep1, rep2, min_dist=0.1):
    pairwise_matrix = torch.norm(rep1 - rep2, dim=-1)
    global a, b, pre_min_dist
    if a is None or pre_min_dist != min_dist:
        pre_min_dist = min_dist
        a, b = find_ab_params(1.0, min_dist)
    # 用umap将距离转换为概率
    similarity_matrix = convert_distance_to_probability(pairwise_matrix, a, b)
    return similarity_matrix, pairwise_matrix


def get_similarity_function(similarity_method):
    if similarity_method == "umap":
        return _umap_similarity
    elif similarity_method == "tsne":
        return _student_t_similarity
    elif similarity_method == "exp":
        return _exp_similarity
    elif similarity_method == "cosine":
        return _cosine_similarity


def _get_correlated_mask(batch_size):
    # 对角矩阵
    diag = np.eye(batch_size)
    # 主对角线以下batch_size
    l1 = np.eye(batch_size, batch_size, k=int(-batch_size / 2))
    # 主对角线以上batch_size
    l2 = np.eye(batch_size, batch_size, k=int(batch_size / 2))
    mask = torch.from_numpy((diag + l1 + l2))
    # 主对角线上，以上batch_size，以下batch_size对应False，通过该mask可以区分所有的similarity pair
    mask = (1 - mask).type(torch.bool)
    # mask = torch.triu(mask)
    return mask


def linear_search(data, target):
    less_index = None
    big_index = None
    for i in range(len(data)):
        if data[i] < target:
            if less_index is None or data[i] > data[less_index]:
                less_index = i
        else:
            if big_index is None or data[i] < data[big_index]:
                big_index = i
    return less_index, big_index


def find_anomalies(data, sigma=2, custom_thresh=None):
    """
    基于3σ原则检测异常点
    :param custom_thresh: 自定义阈值
    :param data: 要求服从或近似正态分布，并且是一维数据
    :param sigma: 允许数值变化的范围
    :return: 异常数据
    """
    if custom_thresh is None:
        mean = np.mean(data)
        std = np.std(data)
        vary_thresh = std * sigma
        lower_limit = mean - vary_thresh
        return data < lower_limit, mean, std
        # upper_limit = mean + vary_thresh
        # flag = np.zeros_like(data).astype(np.bool)
        # flag2 = np.zeros_like(data).astype(np.bool)
        # flag[data < lower_limit] = True
        # flag2[data > upper_limit] = True
        # return np.logical_or(flag, flag2)
    else:
        return data < custom_thresh


def cal_similarity_umap(embeddings, index, neighbor_indices, min_dist=0.1):
    c_data = embeddings[index]
    cur_neighbor_embeddings = embeddings[neighbor_indices]
    cur_dist = np.linalg.norm(cur_neighbor_embeddings - c_data, axis=-1)
    a, b = find_ab_params(1.0, min_dist)
    # 将距离转换为概率
    similarities = convert_distance_to_probability(cur_dist, a, b)
    return similarities


def data_statistics(data, axis):
    data_mean = np.mean(data, axis=axis)
    data_std = np.std(data, axis=axis)
    data_diff = np.max(data, axis=axis) - np.min(data, axis=axis)
    data_change = np.max(data[1:, :] - data[:-1, :], axis=axis)

    norm_data_mean = (data_mean - np.min(data_mean)) / (np.max(data_mean) - np.min(data_mean))
    norm_data_std = (data_std - np.min(data_std)) / (np.max(data_std) - np.min(data_std))
    norm_data_diff = (data_diff - np.min(data_diff)) / (np.max(data_diff) - np.min(data_diff))
    norm_data_change = (data_change - np.min(data_change)) / (np.max(data_change) - np.min(data_change))

    return data_mean, data_std, data_diff, data_change, norm_data_mean, norm_data_std, norm_data_diff, norm_data_change


def get_gong(data, gamma):
    graph = nx.DiGraph()
    graph.add_nodes_from(np.unique(data))
    dists = pdist(data)
    dists = squareform(dists)
    y_dists = (1 - gamma) * dists

    for node_a, row_a in enumerate(data):
        dist_idx = np.argsort(dists[node_a])  # O(nlog n)
        for node_b in dist_idx:
            if node_a == node_b:
                continue

            d_i = y_dists[node_a][node_b]
            first_greater = bisect.bisect_left(dists[node_a][dist_idx], d_i)

            b_is_gong = True

            for node_j in dist_idx[:first_greater]:
                if node_a == node_j:
                    continue

                d_j = y_dists[node_a][node_j]

                if d_j < d_i:
                    b_is_gong = False
                    break  # node_j could be a GONG

            if b_is_gong:
                graph.add_edge(node_a, node_b, weight=dists[node_a][node_b])
    return graph


def get_knng(data, k, max_candidates=60):
    flattened_data = np.reshape(data, (-1, np.product(data.shape[1:])))
    # number of trees in random projection forest
    n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(data.shape[0]))))

    nnd = NNDescent(
        flattened_data,
        n_neighbors=k + 1,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=max_candidates,
        verbose=True
    )
    # 获取近邻点下标和距离
    knn_indices, knn_distances = nnd.neighbor_graph
    knn_indices = knn_indices[:, 1:]
    return knn_indices
