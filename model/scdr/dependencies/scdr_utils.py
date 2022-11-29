import math
import random
import time

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from dataset.warppers import extract_csr, KNNManager
from utils.nn_utils import compute_knn_graph
from utils.umap_utils import fuzzy_simplicial_set_partial


class KeyPointsGenerator:
    RANDOM = "random"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"

    @staticmethod
    def generate(data, key_rate, method=RANDOM, cluster_inner_random=True, prob=None, min_num=0, cover_all=False,
                 **kwargs):
        # cover_all = True：表示将在一次epoch中采样之前的所有数据，每个batch中均包含来自不同聚类的数据
        if key_rate >= 1 or (cover_all and method == KeyPointsGenerator.RANDOM):
            return data, np.arange(0, data.shape[0], 1), None, None
        if method == KeyPointsGenerator.RANDOM:
            return KeyPointsGenerator._generate_randomly(data, key_rate, prob, min_num)
        elif method == KeyPointsGenerator.KMEANS:
            return KeyPointsGenerator._generate_kmeans_based(data, key_rate, cluster_inner_random,
                                                             prob, min_num, cover_all)
        elif method == KeyPointsGenerator.DBSCAN:
            return KeyPointsGenerator._generate_dbscan_based(data, key_rate, cluster_inner_random,
                                                             prob, min_num, cover_all, **kwargs)
        else:
            raise RuntimeError("Unsupported key data generating method. Please ensure that 'method' is one of random/"
                               "kmeans/dbscan.")

    @staticmethod
    def _generate_randomly(data: np.ndarray, key_rate: float, prob=None, min_num=0):
        n_samples = data.shape[0]
        indices = np.arange(0, n_samples, 1)
        key_data_num = max(int(n_samples * key_rate), min_num)
        # TODO: choice这个接口有问题，返回的不是唯一值
        key_indices = np.random.choice(indices, key_data_num, p=prob / np.sum(prob), replace=False)
        return data[key_indices], key_indices, None, None

    @staticmethod
    def _generate_kmeans_based(data, key_rate, is_random=True, prob=None, min_num=0, cover_all=False):
        n_samples = data.shape[0]
        cluster_num = int(math.sqrt(n_samples))
        km = KMeans(n_clusters=cluster_num)
        km.fit(data)
        key_data_num = max(int(n_samples * key_rate), min_num)

        if not cover_all:
            return KeyPointsGenerator._sample_in_each_cluster(data, km.labels_, key_data_num, None, prob, is_random)
        else:
            return KeyPointsGenerator._cover_all_seq(data, km.labels_, key_data_num)

    @staticmethod
    def _generate_dbscan_based(data, key_rate, is_random=True, prob=None, min_num=0, batch_whole=False, **kwargs):
        n_samples = data.shape[0]
        eps = kwargs['eps']
        min_samples = kwargs["min_samples"]
        sta = time.time()
        dbs = DBSCAN(eps, min_samples=min_samples)
        dbs.fit(data)
        # print("cost time:", time.time() - sta)
        key_data_num = max(int(n_samples * key_rate), min_num)

        cal_cluster_acc(dbs.labels_, kwargs["labels"])

        if not batch_whole:
            return KeyPointsGenerator._sample_in_each_cluster(data, dbs.labels_, key_data_num, None, prob, is_random)
        else:
            return KeyPointsGenerator._cover_all_seq(data, dbs.labels_, key_data_num, min_samples * 3)

    @staticmethod
    def _sample_in_each_cluster(data, labels, key_data_num, centroids=None, prob=None, is_random=True):
        key_indices = []
        total_cluster_indices = []
        cluster_indices = []
        n_samples = len(labels)
        key_rate = key_data_num / len(np.argwhere(labels >= 0).squeeze())

        unique_labels = np.unique(labels)
        for item in unique_labels:
            if item < 0:
                continue
            cur_indices = np.argwhere(labels == item).squeeze()
            total_cluster_indices.append(cur_indices)

        if is_random:  # 聚类内部随机采样
            for item in np.unique(labels):
                if item < 0:
                    continue
                cur_indices = np.where(labels == item)[0]
                cur_key_num = int(math.ceil(len(cur_indices) * key_rate))
                cur_prob = None
                if prob is not None:
                    cur_prob = prob[cur_indices]
                    cur_prob /= np.sum(cur_prob)
                sampled_indices = np.random.choice(cur_indices, cur_key_num, p=cur_prob, replace=False)
                cluster_indices.append(np.arange(len(key_indices), len(key_indices) + len(sampled_indices)))
                key_indices.extend(sampled_indices)
        else:  # 聚类内部根据各点到聚类中心的距离进行采样
            if prob is not None:
                raise RuntimeError("Custom sampling probability conflicts with distance-based sampling!")
            for i, item in enumerate(np.unique(labels)):
                if item < 0:
                    continue
                cur_indices = np.where(labels == item)[0]
                cur_center = centroids[i] if centroids is not None else np.mean(data[cur_indices], axis=-1)
                cur_dists = cdist(cur_center, data[cur_indices])
                sorted_indices = np.argsort(cur_dists)
                cur_key_num = int(len(cur_indices) * key_rate)
                sampled_indices = cur_indices[sorted_indices[np.linspace(0, len(cur_indices), cur_key_num, dtype=int)]]
                cluster_indices.append(np.arange(len(key_indices), len(key_indices) + len(sampled_indices)))
                key_indices.extend(sampled_indices)

        exclude_indices = []
        total_indices = np.arange(len(key_indices))
        for item in cluster_indices:
            exclude_indices.append(np.setdiff1d(total_indices, item))

        return data[key_indices], key_indices, cluster_indices, exclude_indices, total_cluster_indices

    @staticmethod
    def _cover_all_seq(data, labels, key_data_num, min_samples=20):
        n_samples = len(np.argwhere(labels >= 0).squeeze())

        key_indices = []
        cluster_indices = []
        exclude_indices = []
        total_cluster_indices = []
        batch_cluster_num = []
        unique_labels = np.unique(labels)
        valid_num = 0
        selected_labels = []
        for item in unique_labels:
            if item < 0:
                continue
            cur_indices = np.argwhere(labels == item).squeeze()
            if len(cur_indices) < min_samples:
                continue
            selected_labels.append(item)
            valid_num += len(cur_indices)
            np.random.shuffle(cur_indices)
            total_cluster_indices.append(cur_indices)
            batch_cluster_num.append(len(cur_indices))

        print("valid num:", valid_num, "total num:", len(labels))

        # 这里的标准是每个batch中只包含 key_data_num个点，然后计算采样所有数据需要的batch数
        batch_num = int(np.floor(valid_num / key_data_num))
        # 每个batch中，各个聚类所包含的点数。聚类规模越大包含的点数越多，这可能就会导致对小规模聚类的各项性能都降低。
        batch_cluster_num = np.floor(np.array(batch_cluster_num) / batch_num).astype(int)
        print("batch_cluster_num", batch_cluster_num)

        for i in range(batch_num):
            cur_k_indices = []
            cur_c_indices = []
            idx = 0
            # print()
            for item in selected_labels:
                num = batch_cluster_num[idx]
                end_idx = min((i + 1) * num, len(total_cluster_indices[idx]))
                select_indices = np.arange(i * num, end_idx)

                if end_idx < (i + 1) * num:  # 需要补齐
                    # print("current len:", end_idx - i*num)
                    left = (i + 1) * num - end_idx
                    select_indices = np.append(select_indices, np.arange(left))
                # print("batch {} cluster {} add {}".format(i, idx, len(select_indices)))

                cur_indices = total_cluster_indices[idx][select_indices]
                cur_c_indices.append(np.arange(len(cur_k_indices), len(cur_k_indices) + len(select_indices)))
                cur_k_indices.extend(cur_indices)
                idx += 1

            key_indices.append(cur_k_indices)
            cluster_indices.append(cur_c_indices)

        for i in range(batch_num):
            cur_e_indices = []
            total_indices = np.arange(len(key_indices[i]))
            for item in cluster_indices[i]:
                cur_e_indices.append(np.setdiff1d(total_indices, item))
            exclude_indices.append(cur_e_indices)

        return None, key_indices, cluster_indices, exclude_indices, total_cluster_indices


class ClusterRepDataSampler:
    def __init__(self, sample_rate=0, min_num=100, cover_all=False):
        self.__sample_rate = sample_rate
        self.__min_num = min_num
        self.__cover_all = cover_all

    def sample(self, fitted_embeddings, eps, min_samples, labels=None):
        _, rep_old_indices, cluster_indices, exclude_indices, total_cluster_indices = \
            KeyPointsGenerator.generate(fitted_embeddings, self.__sample_rate, method=KeyPointsGenerator.DBSCAN,
                                        min_num=self.__min_num, cover_all=self.__cover_all, eps=eps,
                                        min_samples=min_samples, labels=labels)

        if not self.__cover_all:
            rep_old_indices = [rep_old_indices]
            cluster_indices = [cluster_indices]
            exclude_indices = [exclude_indices]

        rep_batch_nums = len(rep_old_indices)

        return rep_batch_nums, rep_old_indices, cluster_indices, exclude_indices, total_cluster_indices

    def dist_to_nearest_cluster_centroids(self, fitted_data, cluster_indices):
        centroids = []
        all_indices = []
        for item in cluster_indices:
            centroids.append(np.mean(fitted_data[item], axis=0))
            all_indices.extend(item)

        dist_matrix = cdist(fitted_data[all_indices], np.array(centroids))
        min_dist = np.min(dist_matrix, axis=1)
        return np.array(centroids), np.mean(min_dist), np.std(min_dist)


class DistributionChangeDetector:
    def __init__(self, lof_based=True):
        self.lof_based = lof_based
        self.lof = None
        self._current_labels = None
        self.acc_list = []
        self.recall_list = []

    def detect_distribution_shift(self, pred_data, re_fit=True, fit_data=None, fit_labels=None, pred_labels=None,
                                  acc=True):
        if acc and re_fit and fit_labels is not None:
            self._gather_distribution(fit_labels)

        if self.lof_based:
            shifted_indices = self._lof_based(pred_data, re_fit, fit_data=fit_data)
        else:
            shifted_indices = None

        if acc:
            self._cal_detect_acc(pred_labels, shifted_indices)

        return shifted_indices

    def _lof_based(self, pred_data, knn_indices, knn_dists, re_fit=True, fit_data=None, n_neighbors=5,
                   contamination=0.1):
        if self.lof is None:
            # self.lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, metric="euclidean",
            #                               contamination=contamination)
            self.lof = MyLocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, metric="euclidean",
                                            contamination=contamination)
        # 每次都需要重新fit，这是非常耗时的。实际上只需要在模型更新之后才需要重新fit。
        if re_fit:
            assert fit_data is not None
            self.lof.fit(fit_data)
        # labels = self.lof.predict(pred_data)
        labels = self.lof.predict_novel(pred_data, knn_indices, knn_dists)
        shifted_indices = np.where(labels == -1)[0]
        return shifted_indices

    def _metric_based(self):
        pass

    def _gather_distribution(self, labels):
        self._current_labels = np.unique(labels)

    def _cal_detect_acc(self, pred_labels, shift_indices):
        true_num = 0
        detected_num = 0
        for i in shift_indices:
            if pred_labels[i] not in self._current_labels:
                true_num += 1
        for i, item in enumerate(pred_labels):
            if item not in self._current_labels and i in shift_indices:
                detected_num += 1
        acc = true_num / len(pred_labels)
        recall = detected_num / len(pred_labels)
        # print(len(shift_indices), acc, recall)
        self.acc_list.append(acc)
        self.recall_list.append(recall)

    def ending(self):
        avg_acc = np.mean(self.acc_list)
        avg_recall = np.mean(self.recall_list)
        print("Avg Acc: %.4f Avg Recall: %.4f" % (avg_acc, avg_recall))


class EmbeddingQualitySupervisor:
    def __init__(self, interval_seconds, manifold_change_num_thresh, bad_embedding_num_thresh, d_scale=None,
                 e_thresh=None, data_reduction="mean", embedding_reduction="mean"):
        self.__last_update_time = None
        # 当新数据到最近流形中心的距离高于d_thresh时，就认为可能来自新的流形。d_thresh可以通过计算模型拟合过的数据到最近流形中心的平均距离得到
        self.__d_scale = d_scale
        # 当新数据的嵌入到k近邻的嵌入的平均距离高于e_thresh时，就认为模型嵌入的质量较差。e_thresh可以通过计算模型拟合过的数据嵌入到k近邻嵌入的平均距离得到
        self.__e_thresh = e_thresh
        self.__interval_seconds = interval_seconds
        # manifold_change_num_thresh应该要小于bad_embedding_num_thresh，因为来自新的流形的数据对嵌入的可信度影响较大
        self.__manifold_change_num_thresh = manifold_change_num_thresh
        self.__bad_embedding_num_thresh = bad_embedding_num_thresh
        self.__new_manifold_data_num = 0
        self.__bad_embedding_data_num = 0
        self._data_reduction = data_reduction
        self._embedding_reduction = embedding_reduction

        # self._lof = LocalOutlierFactor(n_neighbors=10, novelty=True, metric="euclidean",
        #                                contamination=0.1)
        self._lof = MyLocalOutlierFactor(n_neighbors=10, novelty=True, metric="euclidean",
                                         contamination=0.1)

    def update_threshes(self, e_thresh, d_low, d_high):
        self._update_e_thresh(e_thresh)
        self._update_d_scale(d_low, d_high)

    def _update_d_scale(self, low, high):
        low = max(low, 0)
        self.__d_scale = [low, high]

    def _update_e_thresh(self, new_e_thresh):
        self.__e_thresh = new_e_thresh

    def update_model_update_time(self, update_time):
        self.__last_update_time = update_time

    def _judge_model_update(self):
        # 当累积了指定数量的新的流形数据时，或者是累积了指定数量的嵌入质量差的数据时，就更新模型
        update = False
        if self.__last_update_time is not None and time.time() - self.__last_update_time >= self.__interval_seconds:
            update = True
        elif self.__new_manifold_data_num >= self.__manifold_change_num_thresh:
            update = True
        elif self.__bad_embedding_data_num >= self.__bad_embedding_num_thresh:
            update = True

        if update:
            # TODO: 模型更新完成后需要回调update_model_update_time
            self.__new_manifold_data_num = 0
            self.__bad_embedding_data_num = 0

        return update

    def quality_record_lof(self, data, embedding, neighbor_embeddings, knn_indices, knn_dists, pre_data=None):
        manifold_change = False
        need_optimize = False

        assert embedding is not None
        if self._embedding_reduction == "mean":
            embedding_dist = np.mean(cdist(embedding, neighbor_embeddings))
        else:
            embedding_dist = np.max(cdist(embedding, neighbor_embeddings))
        if embedding_dist >= self.__e_thresh:
            self.__bad_embedding_data_num += 1
            need_optimize = True

        if pre_data is not None:
            sta = time.time()
            self._lof.fit(pre_data)
            print("fit time", time.time() - sta)
        # label = self._lof.predict(data)
        label = self._lof.predict_novel(knn_indices.squeeze(), knn_dists.squeeze())
        if label == -1:
            self.__new_manifold_data_num += 1
            manifold_change = True

        return need_optimize, manifold_change, self._judge_model_update()

    def quality_record_2(self, data, embedding, knn_dists, neighbor_embeddings):
        manifold_change = False
        need_optimize = False

        assert embedding is not None
        if self._embedding_reduction == "mean":
            embedding_dist = np.mean(cdist(embedding, neighbor_embeddings))
        else:
            embedding_dist = np.max(cdist(embedding, neighbor_embeddings))
        if embedding_dist >= self.__e_thresh:
            self.__bad_embedding_data_num += 1
            need_optimize = True

        assert data is not None

        if self._data_reduction == "mean":
            data_dist = np.mean(knn_dists)
        else:
            data_dist = np.max(knn_dists)
        # print("====================", data_dist, self.__d_scale)
        if data_dist >= self.__d_scale[1] or data_dist <= self.__d_scale[0]:
            self.__new_manifold_data_num += 1
            manifold_change = True

        return need_optimize, manifold_change, self._judge_model_update()

    def quality_record(self, data, embedding, cluster_centers=None, neighbor_embeddings=None):
        manifold_change = False
        need_optimize = False

        if not manifold_change and neighbor_embeddings is not None:
            assert embedding is not None
            avg_dist = np.mean(cdist(embedding, neighbor_embeddings))
            if avg_dist >= self.__e_thresh:
                self.__bad_embedding_data_num += 1
                need_optimize = True

        if cluster_centers is not None:
            assert data is not None
            min_dist = np.min(cdist(data, cluster_centers))
            # print("min dist:", min_dist)
            if min_dist >= self.__d_scale:
                self.__new_manifold_data_num += 1
                manifold_change = True

        # print("manifold change num: {} bad embedding num: {}".format(self.__new_manifold_data_num, self.__bad_embedding_data_num))
        return need_optimize, manifold_change, self._judge_model_update()


class MyLocalOutlierFactor(LocalOutlierFactor):
    def __init__(
            self,
            n_neighbors=20,
            *,
            algorithm="auto",
            leaf_size=30,
            metric="minkowski",
            p=2,
            metric_params=None,
            contamination="auto",
            novelty=False,
            n_jobs=None,
    ):
        LocalOutlierFactor.__init__(self, n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric, p=p,
                                    metric_params=metric_params, contamination=contamination, novelty=novelty,
                                    n_jobs=n_jobs)
        self._valid_rate = 0.3

    def predict_novel(self, nn_indices, nn_dists):
        valid_indices = np.where(nn_indices < self.n_samples_fit_)[0]
        if len(valid_indices) / self.n_neighbors_ < self._valid_rate:
            return -1

        scores = self.my_score_samples(nn_indices[valid_indices], nn_dists[valid_indices]) - self.offset_

        return -1 if scores < 0 else 1

    def my_score_samples(self, neighbors_indices_X, distances_X):
        dist_k = self._distances_fit_X_[neighbors_indices_X, self.n_neighbors_ - 1]
        reach_dist_array = np.maximum(distances_X, dist_k)

        X_lrd = 1.0 / (np.mean(reach_dist_array) + 1e-10)

        lrd_ratios_array = self._lrd[neighbors_indices_X] / X_lrd

        # as bigger is better:
        return -np.mean(lrd_ratios_array)


def cal_cluster_acc(cluster_labels, gt_labels):
    n_samples = len(gt_labels)
    labels_maps = []
    labels_counts = []
    unique_cluster_labels = np.unique(cluster_labels)
    for idx, item in enumerate(unique_cluster_labels):
        if item < 0:
            labels_maps.append(-1)
            labels_counts.append(0)
            # print("outlier num:", len(item))
            continue

        cur_indices = np.argwhere(cluster_labels == item).squeeze()
        cur_u_labels, cur_u_counts = np.unique(gt_labels[cur_indices], return_counts=True)
        cur_gt_idx = int(np.argmax(cur_u_counts).squeeze())

        true_label = cur_u_labels[cur_gt_idx]
        if true_label in labels_maps:
            i = labels_maps.index(true_label)
            if labels_counts[i] < cur_u_counts[cur_gt_idx]:
                labels_maps[i] = -1
                labels_maps.append(true_label)
                labels_counts.append(cur_u_counts[cur_gt_idx])
            else:
                labels_maps.append(-1)
                labels_counts.append(cur_u_counts[cur_gt_idx])
        else:
            labels_maps.append(true_label)
            labels_counts.append(cur_u_counts[cur_gt_idx])

    # print(labels_maps)
    # print(labels_counts)

    acc_num = 0
    for i in range(len(labels_maps)):
        if labels_maps[i] < 0:
            continue
        acc_num += labels_counts[i]

    print("clustering acc:", acc_num / n_samples)
