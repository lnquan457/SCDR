#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from utils.logger import InfoLogger
from utils.math_utils import *
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
from multiprocessing import Process

from utils.nn_utils import compute_knn_graph, get_pairwise_distance
import seaborn as sns

EPS = 1e-7
np.set_printoptions(suppress=True)


def draw_tmp_2(z, z_2, vis_save_path):
    plt.figure(figsize=(6, 3))
    plt.subplot(121)
    sns.scatterplot(x=z[:, 0], y=z[:, 1], s=10, legend=False, alpha=1.0)
    plt.axis("equal")
    plt.subplot(122)
    sns.scatterplot(x=z_2[:, 0], y=z_2[:, 1], s=10, legend=False, alpha=1.0)
    plt.axis("equal")

    if vis_save_path is not None:
        plt.savefig(vis_save_path, dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def draw_tmp(z_1, z_2, vis_save_path):
    plt.figure(figsize=(4, 4))
    label = np.arange(0, z_1.shape[0], 1)

    plt.scatter(x=z_1[:, 0], y=z_1[:, 1], c=label, cmap='tab20', s=15, alpha=1.0, marker="x")
    plt.scatter(x=z_2[:, 0], y=z_2[:, 1], c=label, cmap='tab20', s=15, alpha=1.0, marker="o")
    plt.axis("equal")

    if vis_save_path is not None:
        plt.savefig(vis_save_path, dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def cal_weighted_shift(old_data_rate, pre_cur_dists, data_nums):
    old_data_rate_matrix = np.repeat(np.expand_dims(old_data_rate, axis=0), data_nums, axis=0)
    weights = (old_data_rate_matrix + old_data_rate_matrix.T) / 2
    triu_indices = np.triu_indices(pre_cur_dists.shape[0], 1)
    weighted_dists = weights[triu_indices] * pre_cur_dists[triu_indices]
    return weighted_dists


def cal_dist_correlation(x, y):
    n_samples = x.shape[0]
    assert x.shape == y.shape
    assert n_samples > 1
    dist_x = cdist(x, x)
    dist_y = cdist(y, y)
    corr_list = []
    for i in range(n_samples):
        corr_list.append(stats.pearsonr(dist_x[0], dist_y[0])[0])
    return corr_list


def metric_mental_map_preservation(cur_embeddings, pre_embeddings, nn_indices=None, k=30):
    pre_n_samples = pre_embeddings.shape[0]
    dists = np.linalg.norm(cur_embeddings[:pre_n_samples] - pre_embeddings, axis=-1) ** 2
    if np.max(dists) == 0:
        return 1, None
    dists /= np.max(dists)
    if nn_indices is None:
        nn_indices = compute_knn_graph(cur_embeddings, None, k, None)[0]

    weights = np.zeros(shape=pre_n_samples)
    for i in range(pre_n_samples):
        weights[i] = 1 - len(np.where(nn_indices[i] >= pre_n_samples)[0]) / len(nn_indices[i])
    dists *= weights
    return 1 - np.mean(dists), nn_indices


def metric_mental_map_preservation_edge(cur_embeddings, pre_embeddings, nn_indices=None, k=30):
    pre_n_samples = pre_embeddings.shape[0]
    if np.max(np.linalg.norm(cur_embeddings[:pre_n_samples] - pre_embeddings, axis=-1)) == 0:
        return 1, None
    dists_cur = cdist(cur_embeddings[:pre_n_samples], cur_embeddings[:pre_n_samples]) ** 2
    dists_pre = cdist(pre_embeddings, pre_embeddings) ** 2
    pre_cur_dists = np.abs(dists_cur - dists_pre)
    pre_cur_dists /= np.max(pre_cur_dists)
    if nn_indices is None:
        nn_indices = compute_knn_graph(cur_embeddings, None, k, None)[0]

    old_data_rate = np.zeros(shape=pre_n_samples)
    for i in range(pre_n_samples):
        old_data_rate[i] = 1 - len(np.where(nn_indices[i] >= pre_n_samples)[0]) / len(nn_indices[i])

    weighted_dists = cal_weighted_shift(old_data_rate, pre_cur_dists, pre_n_samples)
    return 1 - np.mean(weighted_dists), nn_indices


def metric_mental_map_preservation_cntp(cur_embeddings, pre_embeddings):
    n_samples = pre_embeddings.shape[0]
    cluster_num = int(np.sqrt(n_samples))
    km = KMeans(n_clusters=cluster_num)
    cur_labels = km.fit_predict(cur_embeddings)
    cur_centroids = km.cluster_centers_
    pre_labels = km.fit_predict(pre_embeddings)
    pre_centroids = km.cluster_centers_

    intersect_rate = np.zeros(shape=(cluster_num, cluster_num))
    cur_cluster_indices = {}
    pre_cluster_indices = {}
    old_data_rate = np.zeros(shape=cluster_num)

    for i in range(cluster_num):
        cur_cluster_indices[i] = np.where(cur_labels == i)[0]
        pre_cluster_indices[i] = np.where(pre_labels == i)[0]
        old_data_rate[i] = 1 - len(np.where(cur_cluster_indices[i] >= n_samples)[0]) / len(cur_cluster_indices[i])

    corr_cluster = np.zeros(shape=cluster_num, dtype=int)
    for i in range(cluster_num):
        for j in range(i, cluster_num):
            inter_rate = len(np.intersect1d(cur_cluster_indices[i], pre_cluster_indices[j])) / \
                         max(len(cur_cluster_indices[i]), len(pre_cluster_indices[j]))
            intersect_rate[i][j] = inter_rate
            intersect_rate[j][i] = inter_rate
        if np.max(intersect_rate[i]) > 0:
            corr_cluster[i] = np.argmax(intersect_rate[i])
            # 否则说明当前聚类中的所有数据全部是新数据，所以不需要进行对应，即使对应上，后续的权重也是0

    '''
    1. 当前有多个聚类C1，C2，...，Cn对应之前的一个聚类Ck，说明Ck在模型更新后被分成了多个小的聚类，破坏了visual consistency，
       此时这几个聚类离原聚类中心越近说明破坏程度越低，离的越远说明破坏程度越高。
    2. 之前的聚类Ck，在当前没有对应的聚类。说明Ck在模型更新后被完全破坏掉了，破坏了visual consistency。
    '''

    pre_centroids_dists = cdist(pre_centroids, pre_centroids) ** 2
    cur_centroids_dists = cdist(cur_centroids[corr_cluster], cur_centroids[corr_cluster]) ** 2
    pre_cur_dists = np.abs(pre_centroids_dists - cur_centroids_dists)
    pre_cur_dists /= np.max(pre_cur_dists)

    weighted_dists = cal_weighted_shift(old_data_rate, pre_cur_dists, cluster_num)
    return (1 - np.mean(weighted_dists)) * len(np.unique(corr_cluster)) / cluster_num


def metric_visual_consistency_dbscan(cur_embeddings, pre_embeddings, pre_avg_nn_dist, cur_avg_nn_dist, min_samples=10,
                                     sample_ratio=1.0, weighted=True, vc_metric="dist", norm=False,
                                     high_dist2new_data=None, knn_change_indices=None):
    """
        当前的聚类数等于之前的聚类数：计算每对聚类内部的距离变化以及聚类之间的距离变化
        当前的聚类数小于之前的聚类数：说明模型更新将原本的两个或多个聚类合并了，visual consistency被较大程度的破坏
        当前的聚类数大于之前的聚类数：说明模型更新将原本的一个聚类分离成了多个聚类，visual consistency被较大程度的破坏
    """

    # 聚类算法不稳定会造成较大的误差，因为即使投影只发生较小的变化，前后两次的聚类中心可能由于算法误差发生较大的变化
    cur_embeddings, pre_embeddings, cur_cluster_centroids, cur_cluster_indices, pre_cluster_centroids, \
    pre_cluster_indices, noise_factor = cluster_dbscan(cur_avg_nn_dist, cur_embeddings, pre_avg_nn_dist, pre_embeddings,
                                                       min_samples, norm)

    corr_cluster, pre_matched_indices, old_data_rate, intersect_rate = \
        cluster_match(cur_cluster_centroids, cur_cluster_indices, pre_cluster_centroids, pre_cluster_indices,
                      pre_embeddings, weighted, high_dist2new_data, knn_change_indices)

    if vc_metric == "pearson":
        vc = cal_vc_corr(corr_cluster, cur_cluster_centroids, cur_cluster_indices, cur_embeddings,
                         old_data_rate, pre_cluster_centroids, pre_cluster_indices, pre_embeddings, pre_matched_indices,
                         sample_ratio, weighted, noise_factor)
    else:
        vc = cal_vc_dist(corr_cluster, cur_cluster_centroids, cur_cluster_indices, cur_embeddings,
                         old_data_rate, pre_cluster_centroids, pre_cluster_indices, pre_embeddings, pre_matched_indices,
                         sample_ratio, weighted, noise_factor)
    return vc


def cal_vc_dist(corr_cluster, cur_cluster_centroids, cur_cluster_indices, cur_embeddings,
                old_data_rate, pre_cluster_centroids, pre_cluster_indices, pre_embeddings, pre_matched_indices,
                sample_ratio, weighted, noise_factor=0):
    cur_matched_num = len(np.unique(corr_cluster))
    pre_matched_num = len(pre_matched_indices)
    # print("Previous Cluster Matched Num:", pre_matched_num)
    # print("Current Cluster Matched Num:", cur_matched_num)

    dist_change = np.zeros(shape=pre_matched_num)
    cur_matched_data_indices = set()
    pre_matched_data_num = 0
    for i, item in enumerate(pre_matched_indices):
        cur_matched_data_indices = cur_matched_data_indices.union(cur_cluster_indices[corr_cluster[item]])
        pre_matched_data_num += len(pre_cluster_indices[item])

        intersect_indices = np.intersect1d(cur_cluster_indices[corr_cluster[item]], pre_cluster_indices[item])
        sampled_intersect_indices = random.sample(list(intersect_indices),
                                                  max(2, int(math.ceil(len(intersect_indices)) * sample_ratio)))

        inner_dist = np.abs(cur_embeddings[sampled_intersect_indices] - pre_embeddings[sampled_intersect_indices])
        inner_dist = np.mean(inner_dist)
        # print("inner_dist:", inner_dist)

        if pre_matched_num > 1 and cur_matched_num > 1:
            # 可以通过选择多个点来代表一个聚类，以减小聚类算法引入的误差
            pre_dists = np.linalg.norm(pre_cluster_centroids[pre_matched_indices] - pre_cluster_centroids[item],
                                       axis=-1)
            cur_dists = np.linalg.norm(cur_cluster_centroids[corr_cluster[pre_matched_indices]] -
                                       cur_cluster_centroids[corr_cluster[item]], axis=-1)
            outer_dist = np.mean(np.abs(pre_dists - cur_dists))
            # 这里一些相对微小的变化其实是可以忽略的
            # print("outer_dist:", np.abs(pre_dists - cur_dists), outer_dist)
            dists = (inner_dist + outer_dist) / 2
        else:
            dists = inner_dist
            # print("outer_dist:", 0)

        # corr_list[i] = dists * len(pre_cluster_indices[item])
        dist_change[i] = dists
    # corr_list /= pre_matched_data_num
    matched_data_ratio = len(cur_matched_data_indices) / (cur_embeddings.shape[0] - noise_factor)
    # print("Raw Distance Change: %.4f" % np.mean(dist_change), dist_change)
    if weighted:
        old_data_rate **= 2
        dist_change = dist_change * old_data_rate[corr_cluster[pre_matched_indices]]
        # print("Old Data Rate:", old_data_rate[corr_cluster[pre_matched_indices]])
    # print("Weighted Distance Change: %.4f === Matched Data Ratio: %.4f" % (np.mean(dist_change), matched_data_ratio))
    vc = np.mean(dist_change) / matched_data_ratio
    # vc = (1 - np.mean(dist_change)) * matched_data_ratio
    return vc


def cal_vc_corr(corr_cluster, cur_cluster_centroids, cur_cluster_indices, cur_embeddings,
                old_data_rate, pre_cluster_centroids, pre_cluster_indices, pre_embeddings, pre_matched_indices,
                sample_ratio, weighted, noise_factor=0):
    cur_matched_num = len(np.unique(corr_cluster))
    pre_matched_num = len(pre_matched_indices)
    # print("Previous Cluster Matched Num:", pre_matched_num)
    # print("Current Cluster Matched Num:", cur_matched_num)

    corr_list = np.zeros(shape=pre_matched_num)
    cur_matched_data_indices = set()
    pre_matched_data_num = 0
    for i, item in enumerate(pre_matched_indices):
        cur_matched_data_indices = cur_matched_data_indices.union(cur_cluster_indices[corr_cluster[item]])
        pre_matched_data_num += len(pre_cluster_indices[item])

        intersect_indices = np.intersect1d(cur_cluster_indices[corr_cluster[item]], pre_cluster_indices[item])
        sampled_intersect_indices = random.sample(list(intersect_indices),
                                                  max(2, int(math.ceil(len(intersect_indices)) * sample_ratio)))

        inner_corr = cal_dist_correlation(cur_embeddings[sampled_intersect_indices],
                                          pre_embeddings[sampled_intersect_indices])
        inner_corr = np.mean(inner_corr)
        # print("inner_corr:", inner_corr)

        if pre_matched_num > 1 and cur_matched_num > 1:
            pre_dists = np.linalg.norm(pre_cluster_centroids[pre_matched_indices] - pre_cluster_centroids[item],
                                       axis=-1)
            cur_dists = np.linalg.norm(cur_cluster_centroids[corr_cluster[pre_matched_indices]] -
                                       cur_cluster_centroids[corr_cluster[item]], axis=-1)
            outer_corr = stats.pearsonr(pre_dists, cur_dists)[0]
            # print("outer_corr:", outer_corr)
            dists = (inner_corr + outer_corr) / 2
        else:
            dists = inner_corr
            # print("outer_corr:", 0)

        # corr_list[i] = dists * len(pre_cluster_indices[item])
        corr_list[i] = dists
    # corr_list /= pre_matched_data_num
    matched_data_ratio = len(cur_matched_data_indices) / cur_embeddings.shape[0]
    # print("Raw Correlation: %.2f" % np.mean(corr_list), corr_list)
    if weighted:
        old_data_rate **= 2
        selected_old_data_date = old_data_rate[corr_cluster[pre_matched_indices]]
        corr_list = corr_list * selected_old_data_date
        # print("Old Data Rate:", selected_old_data_date)
    # print("Weighted Correlation: %.2f === Matched Data Ratio: %.2f" % (np.mean(corr_list), matched_data_ratio))
    vc = 0.5 * (np.mean(corr_list) + 1) * matched_data_ratio
    return vc


def cluster_match(cur_cluster_centroids, cur_cluster_indices, pre_cluster_centroids, pre_cluster_indices,
                  pre_embeddings, weighted, high_dist2new_data=None, knn_change_indices=None, eta=5):
    pre_n_samples = pre_embeddings.shape[0]
    cur_cluster_num = cur_cluster_centroids.shape[0]
    pre_cluster_num = pre_cluster_centroids.shape[0]
    intersect_rate = np.zeros(shape=(pre_cluster_num, cur_cluster_num))
    # 先前的第i个聚类对应当前的第几个聚类
    corr_cluster = np.ones(shape=pre_cluster_num, dtype=int) * -1
    matched_indices = []
    old_data_rate = np.zeros(shape=cur_cluster_num)
    avg_dist2new_data = np.zeros(shape=cur_cluster_num)
    knn_change_rate = np.zeros(shape=cur_cluster_num)

    for i in range(pre_cluster_num):
        cur_cluster_data_num = len(pre_cluster_indices[i])
        for j in range(cur_cluster_num):
            if i == 0 and weighted:
                # 目前只是计算直接包含新数据的比例，还应该加上与新数据具有邻域关系的节点比例，这些点也会被影响到
                # 1. 可以计算kNN发生改变的节点的比例。但是对PCA等全局降维算法的意义不明确
                # 2. 计算各个聚类中所有（部分）高维数据点到所有最新点的距离均值，然后进行归一化。但是对t-SNE等邻域保持算法的意义不明确
                old_data_rate[j] = len(np.where(cur_cluster_indices[j] < pre_n_samples)[0]) / len(
                    cur_cluster_indices[j])
                if high_dist2new_data is not None:
                    avg_dist2new_data[j] = np.mean(high_dist2new_data[:, cur_cluster_indices[j]])
                if knn_change_indices is not None:
                    knn_change_rate[j] = len(np.intersect1d(cur_cluster_indices[j], knn_change_indices)) / \
                                         len(cur_cluster_indices[i])

            intersect_rate[i][j] = len(np.intersect1d(pre_cluster_indices[i], cur_cluster_indices[j]))

        intersect_rate[i] /= cur_cluster_data_num
        min_rate = eta / cur_cluster_data_num
        if np.max(intersect_rate[i] >= min_rate):
            corr_cluster[i] = np.argmax(intersect_rate[i])
            matched_indices.append(i)

    # print("origin old data rate:", old_data_rate)
    if high_dist2new_data is not None:
        # 这边还有一个问题就是，转换是线性的，
        avg_dist2new_data = (avg_dist2new_data + 1e-8) / (np.max(avg_dist2new_data) + 1e-8)
        old_data_rate *= avg_dist2new_data
        # print("avg_dist2new_data:", avg_dist2new_data)
    if knn_change_indices is not None:
        # old_data_rate *= 1 - knn_change_rate
        # print("knn un-change rate:", 1 - knn_change_rate)
        pass

    return corr_cluster, matched_indices, old_data_rate, intersect_rate


def cluster_dbscan(cur_avg_nn_dist, cur_embeddings, pre_avg_nn_dist, pre_embeddings, min_samples, norm=False):
    db = DBSCAN(eps=pre_avg_nn_dist, min_samples=min_samples)
    pre_labels = db.fit_predict(pre_embeddings)
    db = DBSCAN(eps=cur_avg_nn_dist, min_samples=min_samples)
    cur_labels = db.fit_predict(cur_embeddings)
    cur_cluster_num = len(np.unique(cur_labels)) - (1 if -1 in cur_labels else 0)
    pre_cluster_num = len(np.unique(pre_labels)) - (1 if -1 in pre_labels else 0)
    # print("Previous nn dist:", pre_avg_nn_dist, " Current nn dist:", cur_avg_nn_dist)
    # print("Current Cluster Num: {} Previous Cluster Num: {}".format(cur_cluster_num, pre_cluster_num))
    # print("Current Noise Num:", len(np.where(cur_labels == -1)[0]))
    pre_noise_num = len(np.where(pre_labels == -1)[0])
    # print("Previous Noise Num:", pre_noise_num)
    noise_factor = max(0, pre_noise_num - np.abs(cur_cluster_num - pre_cluster_num))
    cur_cluster_indices = {}
    pre_cluster_indices = {}

    if norm:
        cur_embeddings = my_norm(cur_embeddings)
        pre_embeddings = my_norm(pre_embeddings)

    cur_cluster_centroids = np.empty(shape=(cur_cluster_num, cur_embeddings.shape[1]))
    pre_cluster_centroids = np.empty(shape=(pre_cluster_num, pre_embeddings.shape[1]))
    for i in range(cur_cluster_num):
        cur_cluster_indices[i] = np.where(cur_labels == i)[0]
        cur_cluster_centroids[i] = np.mean(cur_embeddings[cur_cluster_indices[i]], axis=0)
    for i in range(pre_cluster_num):
        pre_cluster_indices[i] = np.where(pre_labels == i)[0]
        pre_cluster_centroids[i] = np.mean(pre_embeddings[pre_cluster_indices[i]], axis=0)
    return cur_embeddings, pre_embeddings, cur_cluster_centroids, cur_cluster_indices, pre_cluster_centroids, \
           pre_cluster_indices, noise_factor


class Metric:
    def __init__(self, dataset_name, origin_data, origin_label, knn_indices, knn_dists,
                 high_dis_matrix=None, result_save_dir=None, val=False, norm=False,
                 subset_indices=None, k=15, method_name=None):

        # 这里的neighbor是用来计算trust、continuity和knn acc的
        self.K = k
        # 在数据子集上做评估
        self.subset_indices = subset_indices
        self.method_name = method_name
        if knn_indices is None or knn_indices.shape[1] < self.K:
            knn_indices, knn_dists = compute_knn_graph(origin_data, None, self.K, None, accelerate=True)

        if norm:
            origin_data = origin_data / 255
        self.dataset_name = dataset_name
        self.origin_data = origin_data
        self.result_save_dir = result_save_dir
        self.val = val

        self.flattened_data = self.origin_data
        if len(self.flattened_data.shape) > 2:
            self.flattened_data = np.reshape(self.flattened_data, (self.flattened_data.shape[0],
                                                                   np.product(self.flattened_data.shape[1:])))

        self.val_count = 0
        self.origin_label = origin_label
        self.embedding_data = None
        self.n_samples = self.metric_dc_num_samples()

        self.knn_indices = knn_indices.astype(np.int)
        self.knn_dists = knn_dists

        if high_dis_matrix is not None:
            if subset_indices is None:
                self.high_dis_matrix = high_dis_matrix
            else:
                self.high_dis_matrix = high_dis_matrix[subset_indices]
                self.high_dis_matrix = self.high_dis_matrix[:, subset_indices]
        else:
            self.high_dis_matrix = get_pairwise_distance(origin_data, "euclidean", None)

        self.high_nn_indices = np.argsort(self.high_dis_matrix, axis=1)

        self.low_dis_matrix = None
        self.low_knn_indices = None

        # 是否对丢失邻居点分布进行记录
        self.record_lost_neighbors = False
        # 丢失的邻居秩序作为键，值表示个数，用于绘制邻居点的保持情况
        self.lost_neighbor_dict = {}
        self.fake_neighbor_dict = {}
        self.lost_neighbor_points = []
        self.fake_neighbor_points = []
        self.first_lost_neighbor_dict = {}
        self.first_fake_neighbor_dict = {}

        self.detail_trust_cont = True
        # 虚假邻居中，与样本类别相同的比例
        self.sim_fake_num = 0
        # 丢失邻居中，与样本类别不同的比例， 两者都是越高越好
        self.dissim_lost_num = 0

        # 计算对称化所产生的邻居的秩序分布
        # if self.origin_sym_knn_indices is not None:
        #     self.metric_symmetry_neighbor_distribution()

        # 删除：noise evaluation, coranking matrix, neighbor_indices, noise neighbor hit, fake neighbor distribution,
        # neighbor rank distribution, symmetry_neighbor_distribution, noise_hold, noise_rank

    def cal_all_metrics(self, k, embedding_data, knn_k=10, compute_shepard=False, final=False):

        if self.subset_indices is not None:
            embedding_data = embedding_data[self.subset_indices]

        self.val_count += 1
        self.acquire_low_distance(embedding_data)
        # normalized_stress = self.metric_normalized_stress(embedding_data)
        trust = self.metric_trustworthiness(k, embedding_data, final=final)
        continuity = self.metric_continuity(k, embedding_data, final)

        neighbor_hit = self.metric_neighborhood_hit(k, embedding_data)

        # shepard_corr = 0
        # if compute_shepard:
        #     shepard_corr = self.metric_shepard_diagram_correlation(embedding_data)

        knn_ac = knn_score(embedding_data, self.origin_label, knn_k, pair_distance=self.low_dis_matrix)
        # knn_ac_5 = knn_score(embedding_data, self.origin_label, 5, pair_distance=self.low_dis_matrix)
        # knn_ac_1 = knn_score(embedding_data, self.origin_label, 1, pair_distance=self.low_dis_matrix)
        # knn_ac_5 = self.metric_avg_fn_rank(embedding_data)
        # knn_ac_1 = 0
        sc = metric_silhouette_score(embedding_data, self.origin_label)
        dsc = self.metric_dsc(embedding_data)
        # gon = self.metric_gong(embedding_data)
        gon = 0

        self.embedding_data = None
        self.low_dis_matrix = None
        self.low_knn_indices = None
        return trust, continuity, neighbor_hit, knn_ac, sc, dsc

    def eval_dataset(self, k):
        is_balanced, class_num, class_counts = self.metric_dc_dataset_is_balanced()
        sparsity_ratio = self.metric_dc_sparsity_ratio()
        intrinsic_dim = self.metric_dc_intrinsic_dim()
        origin_neighbor_hit = self.metric_origin_neighbor_hit(k)

        cls_count_str = np.array2string(class_counts, separator="/")[1:-1]
        cls_count_str = cls_count_str.replace(" ", "")
        return class_num, cls_count_str, is_balanced, sparsity_ratio, intrinsic_dim, origin_neighbor_hit, \
               self.n_samples, self.origin_data.shape[1]

    def metric_dc_dataset_is_balanced(self):
        counts = np.zeros((len(np.unique(self.origin_label)),))
        classes, class_counts = np.unique(self.origin_label, return_counts=True)
        for i, l in enumerate(np.unique(self.origin_label)):
            counts[i] = np.count_nonzero(self.origin_label == l)

        return np.min(counts) / np.max(counts) > 0.5, len(classes), class_counts

    # 获取数据集规模
    def metric_dc_num_samples(self):
        return self.origin_data.shape[0]

    # 获取数据集维数
    def metric_dc_num_features(self):
        return self.origin_data.shape[1]

    # 获取数据集类别数
    def metric_dc_num_classes(self):
        return len(np.unique(self.origin_label))

    def metric_dc_sparsity_ratio(self):
        """
        返回数据集self.origin_data的稀疏程度，N表示数据集样本数，n表示数据维度，k表示非0值
        :param self.origin_data: 数据集
        :return: 稀疏程度 sr = k/(n*N)
        """
        return 1.0 - (np.count_nonzero(self.flattened_data) /
                      float(self.flattened_data.shape[0] * self.flattened_data.shape[1]))

    def metric_dc_intrinsic_dim(self):
        """
        利用PCA计算数据集self.origin_data的本质维度
        :param self.origin_data: 数据集
        :return: 利用PCA算法计算得到的主成分和大于等于0.95所需的主成分数
        """
        pca = PCA()
        pca.fit(self.flattened_data)
        return np.where(pca.explained_variance_ratio_.cumsum() >= 0.95)[0][0] + 1

    def metric_origin_neighbor_hit(self, k):

        def cal(indices=None):
            t_knn_labels = origin_knn_labels if indices is None else origin_knn_labels[indices]
            t_origin_labels = origin_labels if indices is None else origin_labels[indices]
            nn_label_same_ratio = np.mean(np.mean((t_knn_labels == np.tile(t_origin_labels.reshape((-1, 1)), k))
                                                  .astype('uint8'), axis=1))
            return nn_label_same_ratio

        origin_knn_labels = np.zeros_like(self.knn_indices)
        origin_labels = self.origin_label
        for i in range(self.metric_dc_num_samples()):
            origin_knn_labels[i] = self.origin_label[self.knn_indices[i].astype(np.int)]

        # c3_indices = np.argwhere(self.origin_label == 3).squeeze()
        # c4_indices = np.argwhere(self.origin_label == 4).squeeze()
        # onh_3 = cal(c3_indices)
        # onh_4 = cal(c4_indices)
        # print("cluster 3 nh: {} cluster 4 nh: {}".format(onh_3, onh_4))

        single_check_cls = np.unique(self.origin_label)
        for cls in single_check_cls:
            cls_indices = np.argwhere(self.origin_label == cls).squeeze()
            onh = cal(cls_indices)
            print("Cluster %d Neighbor Hit = %.4f" % (cls, onh))

        origin_nn_label_same_ratio = cal()
        return origin_nn_label_same_ratio

    def metric_neighborhood_hit(self, k, embedding_data):
        # 获取原数据近邻点下标和距离
        if self.low_knn_indices is None:
            self.acquire_low_distance(embedding_data)

        pred_knn_labels = np.zeros_like(self.low_knn_indices)
        for i in range(self.low_knn_indices.shape[0]):
            pred_knn_labels[i] = self.origin_label[self.low_knn_indices[i]]

        nn_label_same_ratio = np.mean(
            np.mean((pred_knn_labels == np.tile(self.origin_label.reshape((-1, 1)), k)).astype('uint8'), axis=1))
        return nn_label_same_ratio

    def acquire_low_distance(self, embedding_data):
        embedding_data = np.reshape(embedding_data, (-1, np.product(embedding_data.shape[1:])))
        if self.low_dis_matrix is None:
            self.low_dis_matrix = get_pairwise_distance(embedding_data, metric="euclidean")
        if self.low_knn_indices is None:
            self.low_knn_indices = np.argsort(self.low_dis_matrix, axis=1)[:, 1:self.K + 1]

    def metric_fake_lost(self, embedding_data, k, high_nn_indices=None):
        self.acquire_low_distance(embedding_data)
        # 按距离从小到大进行排序，默认按列进行排序
        # 筛选出高维空间中每个数据点的除自己以外的前k个邻居

        if high_nn_indices is None:
            high_nn_indices = self.high_nn_indices

        knn_indices = high_nn_indices[:, 1:k + 1]

        n = knn_indices.shape[0]
        fake_num = 0
        sim_fake_num = 0
        lost_num = 0
        dissim_lost_num = 0

        for i in range(n):
            # 这里的V代表的是在高维空间中但是不在低维空间上的邻居点，也就是低维空间上丢失的邻居点
            V = np.setdiff1d(knn_indices[i], self.low_knn_indices[i])
            # 返回在knn_proj（低维空间）中但是不在knn_orig（高维空间中）的排序后的邻居索引
            U = np.setdiff1d(self.low_knn_indices[i], knn_indices[i])

            lost_num += len(V)
            dissim_lost_num += len(np.where(self.origin_label[i] != self.origin_label[V])[0])
            fake_num += len(U)
            sim_fake_num += len(np.where(self.origin_label[i] == self.origin_label[U])[0])

        self.embedding_data = None
        self.low_dis_matrix = None
        self.low_knn_indices = None
        return sim_fake_num / fake_num if fake_num > 0 else -1, dissim_lost_num / lost_num if lost_num > 0 else -1

    def metric_sim_fake_dissim_lost(self, embedding_data):
        self.acquire_low_distance(embedding_data)
        knn_indices = self.knn_indices
        n = embedding_data.shape[0]
        sim_fake_num = 0
        dissim_lost_num = 0
        for i in range(n):
            # 返回在knn_proj（低维空间）中但是不在knn_orig（高维空间中）的排序后的邻居索引
            U = np.setdiff1d(self.low_knn_indices[i], knn_indices[i])
            V = np.setdiff1d(knn_indices[i], self.low_knn_indices[i])
            dissim_lost_num += len(np.where(self.origin_label[i] != self.origin_label[V])[0])
            sim_fake_num += len(np.where(self.origin_label[i] == self.origin_label[U])[0])
        self.low_knn_indices = None
        self.low_dis_matrix = None
        return sim_fake_num, dissim_lost_num

    """
    trustworthiness度量的是虚假邻居点比例，即在低维空间中点i的邻居点，但是不属于原高维空间中点i的邻居点的比例，
    0表示低维空间中所有的邻居点都不是原高维空间中数据i的邻居点，而1则相反。
    这个值较小就说明算法把原来高维空间中相距较远的两个点投影到比较近的距离了。
    """

    def metric_trustworthiness(self, k, embedding_data, high_nn_indices=None, final=False):
        self.acquire_low_distance(embedding_data)
        # 按距离从小到大进行排序，默认按列进行排序
        # 筛选出高维空间中每个数据点的除自己以外的前k个邻居

        if high_nn_indices is None:
            high_nn_indices = self.high_nn_indices

        knn_indices = self.knn_indices

        sum_i = 0
        n = knn_indices.shape[0]
        fake_num = 0
        sim_fake_num = 0
        for i in range(n):
            # 返回在knn_proj（低维空间）中但是不在knn_orig（高维空间中）的排序后的邻居索引
            U = np.setdiff1d(self.low_knn_indices[i], knn_indices[i])
            if self.detail_trust_cont:
                fake_num += len(U)
                sim_fake_num += len(np.where(self.origin_label[i] == self.origin_label[U])[0])
            sum_j = 0
            for j in range(U.shape[0]):
                high_rank = np.where(high_nn_indices[i] == U[j])[0][0]
                sum_j += high_rank - k

                # ==============================记录虚假邻居点的秩序分布=============================
                # 低维空间上的秩序
                if self.record_lost_neighbors:
                    low_rank = np.where(self.low_knn_indices[i] == U[j])[0][0]
                    if low_rank not in self.fake_neighbor_dict:
                        self.fake_neighbor_dict[low_rank] = 0
                    self.fake_neighbor_dict[low_rank] += 1
                    self.fake_neighbor_points.append([low_rank, high_rank])
                # =================================================================================

            sum_i += sum_j

        if fake_num != 0:
            self.sim_fake_num = sim_fake_num

        return 1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)

    """
    continuity测量的是在低维空间样本点i中对于高维空间中样本点i的邻居点的保留比例，与之前的trustworthiness
    相对应，1代表所有邻居点全部保留，0代表所有原高维空间中的邻居点全部丢失。
    这个值较小就说明算法把原来高维空间中相距很近的两个点投影到了比较远的距离。
    """

    def metric_continuity(self, k, embedding_data, final=False, high_nn_indices=None):
        self.acquire_low_distance(embedding_data)

        if high_nn_indices is None:
            high_nn_indices = self.high_nn_indices

        knn_indices = self.knn_indices
        n = knn_indices.shape[0]

        sum_i = 0
        lost_num = 0
        dissim_lost_num = 0

        for i in range(n):
            # 这里的V代表的是在高维空间中但是不在低维空间上的邻居点，也就是低维空间上丢失的邻居点
            V = np.setdiff1d(knn_indices[i], self.low_knn_indices[i])

            if self.detail_trust_cont:
                lost_num += len(V)
                dissim_lost_num += len(np.where(self.origin_label[i] != self.origin_label[V])[0])

            pro_nn_indices = np.argsort(self.low_dis_matrix[i])

            sum_j = 0
            for j in range(V.shape[0]):
                # 丢失邻居点在低维空间上的秩序
                low_rank = np.where(pro_nn_indices == V[j])[0]

                # ==============================记录丢失邻居点的秩序分布=============================
                # 高维空间上的秩序
                high_rank = np.where(knn_indices[i] == V[j])[0][0]
                if high_rank not in self.lost_neighbor_dict:
                    self.lost_neighbor_dict[high_rank] = 0
                self.lost_neighbor_dict[high_rank] += 1
                self.lost_neighbor_points.append([high_rank, low_rank])
                # =================================================================================

                sum_j += low_rank - k

            sum_i += sum_j

        if lost_num != 0:
            self.dissim_lost_num = dissim_lost_num

        if self.val_count == 1:
            self.first_lost_neighbor_dict = dict.copy(self.lost_neighbor_dict)

        return 1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)

    """
    高维空间和低维空间中相同数据点对之间的距离的平均值，越小越好
    """

    def metric_normalized_stress(self, embedding_data):
        self.acquire_low_distance(embedding_data)
        # 归一化
        high_dis = self.high_dis_matrix[np.triu_indices(self.high_dis_matrix.shape[0])]
        low_dis = self.low_dis_matrix[np.triu_indices(self.low_dis_matrix.shape[0])]
        return np.sum((high_dis - low_dis) ** 2) / np.sum(high_dis ** 2)
        # return np.sum(np.square(normalized_high_dis - normalized_low_dis))

    """
    通过Shepard Diagram来表示高维空间和低维空间中的数据对距离的关系，理想状态是均分部在
    主对角线上，即高度线性相关，通过斯皮尔曼相关系数来描述，斯皮尔曼相关系数是基于每个变量的秩序值
    """

    def metric_shepard_diagram_correlation(self, embedding_data, high_dis_matrix=None):
        self.acquire_low_distance(embedding_data)

        if high_dis_matrix is None:
            high_dis_matrix = self.high_dis_matrix

        # f_high_dis = self.high_dis_matrix[np.triu_indices(self.high_dis_matrix.shape[0], 1)]
        # f_low_dis = self.low_dis_matrix[np.triu_indices(self.low_dis_matrix.shape[0], 1)]
        # return stats.spearmanr(f_high_dis,f_low_dis)[0]

        # 耗费很大的内存空间
        # corr, _ = stats.spearmanr(self.high_dis_matrix, self.low_dis_matrix)
        corr_list = []
        for i in range(high_dis_matrix.shape[0]):
            corr, _ = stats.spearmanr(high_dis_matrix[i], self.low_dis_matrix[i])
            corr_list.append(corr)
        return np.mean(corr_list)

    def metric_dsc(self, embedding_data):
        classes = np.unique(self.origin_label.astype(int))
        cls_centroids = []
        for cls in classes:
            indices = np.argwhere(self.origin_label == cls).squeeze()
            centroid = np.mean(embedding_data[indices], axis=0)
            cls_centroids.append(centroid)
        flags = np.ones(self.n_samples)
        diff = np.repeat(np.expand_dims(embedding_data, axis=1), len(classes), 1) - \
               np.repeat(np.expand_dims(np.array(cls_centroids), axis=0), self.n_samples, 0)
        dist = np.linalg.norm(diff, axis=-1)
        closet_cls_indices = np.argmin(dist, axis=1).astype(np.int)
        closet_cls = classes[closet_cls_indices]
        dsc = np.mean(closet_cls == self.origin_label)
        return dsc

    def metric_gong(self, embedding_data, gamma=0.35, k=2):
        n_samples = embedding_data.shape[0]
        thresh_num = 5000
        if n_samples > thresh_num:
            neighbor_graph = get_knng(embedding_data, k)
        else:
            neighbor_graph = get_gong(embedding_data, gamma).neighbors

        cp = []
        for node in range(n_samples):
            if n_samples > thresh_num:
                neighbor_indices = list(neighbor_graph[node])
            else:
                neighbor_indices = list(neighbor_graph(node))
            neighbour_classes = list(self.origin_label[neighbor_indices])
            amount_of_neighbours = len(neighbour_classes)

            self_class = self.origin_label[node]
            count_of_self = neighbour_classes.count(self_class)

            if amount_of_neighbours == 0:
                cp.append(1)
            else:
                cp.append(count_of_self / amount_of_neighbours)

        cp = np.array(cp)
        classes = np.unique(self.origin_label)
        cpt = []
        for cls in classes:
            cur_indices = np.argwhere(self.origin_label == cls).squeeze()
            cpt.append(np.mean(cp[cur_indices]))

        return np.mean(cpt)


class MetricProcess(Process, Metric):
    def __init__(self, queue_set, message_queue, dataset_name, origin_data, origin_label, knn_indices,
                 knn_dist, high_dis_matrix=None, result_save_dir=None, val=False, norm=True,
                 subset_indices=None, k=10, method_name=None):
        self.name = "测试集-评估进程" if val else "训练集-评估进程"
        Process.__init__(self, name=self.name)
        Metric.__init__(self, dataset_name, origin_data, origin_label, knn_indices, knn_dist,
                        high_dis_matrix, result_save_dir, val, norm,
                        subset_indices, k, method_name)
        self.queue_set = queue_set
        self.message_queue = message_queue

    def run(self) -> None:
        while True:
            # 从管道中接收评估数据
            if self.val:
                epoch, k, embedding_data, final, noise_test = self.queue_set.test_eval_data_queue.get()
            else:
                epoch, k, embedding_data, final, noise_test = self.queue_set.eval_data_queue.get()

            trust, continuity, neighbor_hit, knn_ac, sc, dsc = self.cal_all_metrics(k, embedding_data, final=final)
            metric_template = "Trust: %.4f Continuity: %.4f Neighbor Hit: %.4f KA: %.4f SC: %.4f DSC: %.4f"
            metric_output = metric_template % (trust, continuity, neighbor_hit, knn_ac, sc, dsc)

            if self.val:
                metric_output = "Validation " + metric_output

            InfoLogger.info(metric_output)
            self.message_queue.put(metric_output)

            res = [trust.item(), continuity.item(), neighbor_hit, knn_ac, sc, dsc]

            # 返回评估结果到管道中

            if self.val:
                self.queue_set.test_eval_result_queue.put(res)
            else:
                self.queue_set.eval_result_queue.put(res)

            if final:
                break


def knn_score(data, labels_gt, k, metric="euclidean", pair_distance=None):
    """
        预测函数
        :param k: 近邻数
        :param metric: 距离度量方式
        :param pair_distance: 待预测数据集距离对
        :param labels_gt: 待预测数据集标签
        :param data: 待预测数据集
        :return: 预测精度
        """
    n_samples = data.shape[0]
    if pair_distance is None:
        flattened_data = np.reshape(data, (n_samples, np.prod(data.shape[1:])))
        pair_distance = get_pairwise_distance(flattened_data, metric, preload=False)
    labels_predict = []
    for i in range(n_samples):
        nearest = np.argsort(pair_distance[i])
        top_k = [labels_gt[i] for i in nearest[1:k + 1]]
        nums, counts = np.unique(top_k, return_counts=True)
        labels_predict.append(nums[np.argmax(counts)])
    labels_predict = np.array(labels_predict)
    labels_gt = labels_gt[:n_samples]
    acc = np.sum(labels_predict == labels_gt) / len(labels_gt)
    return acc


def metric_nmi_and_silhouette_score(data, label_gt, n_init=10):
    unique_cls = list(np.unique(label_gt))
    processed_gt_labels = []
    n_classes = len(unique_cls)
    estimator = KMeans(n_clusters=n_classes, n_init=n_init)  # 构造聚类器
    estimator.fit(data)  # 聚类
    label_pred = estimator.labels_
    # 将kmeans算法产生的label和真实label对齐
    label_align = np.zeros((n_classes, n_classes))

    for i in range(data.shape[0]):
        knn_label = label_pred[i]
        real_label = unique_cls.index(label_gt[i])
        processed_gt_labels.append(real_label)
        label_align[knn_label, real_label] += 1
    label_align = np.argmax(label_align, axis=1)
    for i in range(data.shape[0]):
        label_pred[i] = label_align[label_pred[i]]
    accuracy = np.mean(processed_gt_labels == label_pred)

    return accuracy, metrics.normalized_mutual_info_score(processed_gt_labels, label_pred), \
           metrics.silhouette_score(data, processed_gt_labels)


def metric_silhouette_score(data, labels):
    return metrics.silhouette_score(data, np.array(labels, dtype=int), metric='euclidean')


def my_norm(data):
    min_s = np.expand_dims(np.min(data, axis=0), axis=0)
    max_s = np.expand_dims(np.max(data, axis=0), axis=0)
    data = (data - min_s) / (max_s - min_s)
    return data
