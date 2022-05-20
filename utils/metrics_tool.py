#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from utils.logger import InfoLogger
from utils.math_utils import *
from scipy import stats
from sklearn.cluster import KMeans
from sklearn import metrics
from multiprocessing import Process

from utils.nn_utils import compute_knn_graph, get_pairwise_distance


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

    def metric_mental_map_preservation(self, cur_embeddings, pre_embeddings):
        pass



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
        top_k = [labels_gt[i] for i in nearest[1:k+1]]
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
