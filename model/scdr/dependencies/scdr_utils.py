import math
import random
import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor

from dataset.warppers import extract_csr
from utils.metrics_tool import metric_neighbor_preserve_introduce
from utils.nn_utils import compute_knn_graph
from utils.umap_utils import fuzzy_simplicial_set_partial


class KeyPointsGenerator:

    RANDOM = "random"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"

    @staticmethod
    def generate(data, key_rate, method=RANDOM, cluster_inner_random=True, prob=None, min_num=0, **kwargs):
        if key_rate >= 1:
            return data, np.arange(0, data.shape[0], 1)
        if method == KeyPointsGenerator.RANDOM:
            return KeyPointsGenerator._generate_randomly(data, key_rate, prob, min_num)
        elif method == KeyPointsGenerator.KMEANS:
            return KeyPointsGenerator._generate_kmeans_based(data, key_rate, cluster_inner_random,
                                                                              prob, min_num)
        elif method == KeyPointsGenerator.DBSCAN:
            return KeyPointsGenerator._generate_dbscan_based(data, key_rate, cluster_inner_random,
                                                                              prob, min_num, **kwargs)
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
        return data[key_indices], key_indices

    @staticmethod
    def _generate_kmeans_based(data: np.ndarray, key_rate: float, is_random=True, prob=None, min_num=0, **kwargs):
        n_samples = data.shape[0]
        cluster_num = int(math.sqrt(n_samples))
        km = KMeans(n_clusters=cluster_num)
        km.fit(data)
        key_data_num = max(int(n_samples * key_rate), min_num)

        key_data, key_indices, cluster_indices, exclude_indices = \
            KeyPointsGenerator._sample_in_each_cluster(data, km.labels_, key_data_num, None, prob, is_random)
        return key_data, key_indices, cluster_indices, exclude_indices

    @staticmethod
    def _generate_dbscan_based(data, key_rate, is_random=True, prob=None, min_num=0, **kwargs):
        n_samples = data.shape[0]
        eps = kwargs['eps']
        min_samples = kwargs["min_samples"]
        dbs = DBSCAN(eps, min_samples=min_samples)
        dbs.fit(data)
        key_data_num = max(int(n_samples * key_rate), min_num)
        key_data, key_indices, cluster_indices, exclude_indices = \
            KeyPointsGenerator._sample_in_each_cluster(data, dbs.labels_, key_data_num, None, prob, is_random)
        return key_data, key_indices, cluster_indices, exclude_indices

    @staticmethod
    def _sample_in_each_cluster(data, labels, key_data_num, centroids=None, prob=None, is_random=True):
        key_indices = []
        cluster_indices = []
        n_samples = len(labels)
        key_rate = key_data_num / len(np.argwhere(labels >= 0).squeeze())
        if is_random:  # 聚类内部随机采样
            for item in np.unique(labels):
                if item < 0:
                    continue
                cur_indices = np.where(labels == item)[0]
                cur_key_num = int(len(cur_indices) * key_rate)
                cur_prob = None
                if prob is not None:
                    cur_prob = prob[cur_indices]
                    cur_prob /= np.sum(cur_prob)
                sampled_indices = np.random.choice(cur_indices, cur_key_num, p=cur_prob, replace=False)
                cluster_indices.append(np.arange(len(key_indices), len(key_indices)+len(sampled_indices)))
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
                cluster_indices.append(np.arange(len(key_indices), len(key_indices)+len(sampled_indices)))
                key_indices.extend(sampled_indices)

        exclude_indices = []
        total_indices = np.arange(len(key_indices))
        for item in cluster_indices:
            exclude_indices.append(np.setdiff1d(total_indices, item))

        return data[key_indices], key_indices, cluster_indices, exclude_indices


class RepDataSampler:
    def __init__(self, n_neighbors, sample_rate, minimum_sample_data_num, time_based_sample=False, metric_based_sample=False):
        self.time_based_sample = time_based_sample
        self.metric_based_sample = metric_based_sample
        self.n_neighbors = n_neighbors
        self.sample_rate = sample_rate
        self.minimum_sample_data_num = minimum_sample_data_num

        self.time_weights = None
        self.min_weight = 0.1
        self.decay_rate = 0.9
        self.decay_iter_time = 10
        self.pre_update_time = None

    def update_sample_weight(self, n_samples):
        if not self.time_based_sample:
            return

        # 如果速率产生的很快，那么采样的概率也会衰减的很快，所以这里的更新应该要以时间为标准
        if self.time_weights is None:
            self.time_weights = np.ones(shape=n_samples)
            self.pre_update_time = time.time()
        else:
            cur_time = time.time()
            if cur_time - self.pre_update_time > self.decay_iter_time:
                self.time_weights *= self.decay_rate
                self.time_weights[self.time_weights < self.min_weight] = self.min_weight
                self.pre_update_time = cur_time
            self.time_weights = np.concatenate([self.time_weights, np.ones(shape=n_samples)])

    def sample_training_data(self, pre_data, pre_embeddings, high_knn_indices, must_indices, total_data_num):
        # 采样训练子集
        pre_n_samples = pre_embeddings.shape[0]
        if self.sample_rate >= 1:
            return np.arange(0, total_data_num, 1)

        sampled_indices = None
        prob = None
        if self.time_based_sample and self.metric_based_sample:
            prob = np.copy(self.time_weights)[:pre_n_samples]
            if self.metric_based_sample:
                neighbor_lost_weights = self._cal_neighbor_lost_weights(high_knn_indices, pre_embeddings)
                # 如何在两者之间进行权衡
                prob = (neighbor_lost_weights + prob) / 2
        elif self.time_based_sample:
            prob = self.time_weights[:pre_n_samples]
        elif self.metric_based_sample:
            neighbor_lost_weights = self._cal_neighbor_lost_weights(high_knn_indices, pre_embeddings)
            prob = neighbor_lost_weights
        else:
            sampled_indices = self._random_sample(pre_n_samples)

        if sampled_indices is None:
            sampled_indices = KeyPointsGenerator.generate(pre_data, self.sample_rate, False, prob=prob,
                                                          min_num=self.minimum_sample_data_num)[1]
        if must_indices is not None:
            sampled_indices = np.union1d(sampled_indices, must_indices).astype(int)

        return sampled_indices

    def _cal_neighbor_lost_weights(self, high_knn_indices, embeddings, alpha=2):
        pre_n_samples = embeddings.shape[0]
        high_knn_indices = high_knn_indices[:pre_n_samples]
        # TODO：这里需要进一步提高效率
        low_knn_indices = compute_knn_graph(embeddings, None, self.n_neighbors, None)[0]
        # 这里还应该有一个阈值，只需要大于这个阈值一定程度便可以视为较好了
        preserve_rate = metric_neighbor_preserve_introduce(low_knn_indices, high_knn_indices)[1]
        # mean, std = np.mean(preserve_rate), np.std(preserve_rate)
        # preserve_rate[preserve_rate > mean + alpha * std] = 1
        return (1 - preserve_rate) ** 2

    def _random_sample(self, pre_n_samples):
        sampled_num = max(int(pre_n_samples * self.sample_rate), self.minimum_sample_data_num)
        all_indices = np.arange(0, pre_n_samples, 1)
        np.random.shuffle(all_indices)
        sampled_indices = all_indices[:sampled_num]
        return sampled_indices


class DistributionChangeDetector:
    def __init__(self, lof_based=True):
        self.lof_based = lof_based
        self.lof = None
        self._current_labels = None
        self.acc_list = []
        self.recall_list = []

    def detect_distribution_shift(self, pred_data, re_fit=True, fit_data=None, fit_labels=None, pred_labels=None, acc=True):
        if acc and re_fit and fit_labels is not None:
            self._gather_distribution(fit_labels)

        if self.lof_based:
            shifted_indices = self._lof_based(pred_data, re_fit, fit_data=fit_data)
        else:
            shifted_indices = None

        if acc:
            self._cal_detect_acc(pred_labels, shifted_indices)

        return shifted_indices

    def _lof_based(self, pred_data, re_fit=True, fit_data=None, n_neighbors=5, contamination=0.1):
        if self.lof is None:
            self.lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, metric="euclidean",
                                          contamination=contamination)
        # 每次都需要重新fit，这是非常耗时的。实际上只需要在模型更新之后才需要重新fit。
        if re_fit:
            assert fit_data is not None
            self.lof.fit(fit_data)
        labels = self.lof.predict(pred_data)
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


class StreamingDataRepo:
    # 主要负责记录当前的流数据，以及更新它们的k近邻
    def __init__(self, n_neighbors):
        self.n_neighbor = n_neighbors
        self.total_data = None
        self.total_label = None
        self.total_embeddings = None

    def get_n_samples(self):
        return self.total_data.shape[0] if self.total_data is not None else 0

    def add_new_data(self, data=None, embeddings=None, labels=None):
        if data is not None:
            if self.total_data is None:
                self.total_data = data
            else:
                self.total_data = np.concatenate([self.total_data, data], axis=0)

        if embeddings is not None:
            if self.total_embeddings is None:
                self.total_embeddings = embeddings
            else:
                self.total_embeddings = np.concatenate([self.total_embeddings, embeddings], axis=0)

        if labels is not None:
            if self.total_label is None:
                self.total_label = np.array(labels)
            else:
                if isinstance(labels, list):
                    self.total_label = np.concatenate([self.total_label, labels])
                else:
                    self.total_label = np.append(self.total_label, labels)


