import math
import random

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


class KeyPointsGenerater:

    @staticmethod
    def generate(data, key_rate, is_random=False, cluster_inner_random=True, prob=None, min_num=0):
        if is_random:
            key_data, key_indices = KeyPointsGenerater._generate_randomly(data, key_rate, prob, min_num)
        else:
            key_data, key_indices = KeyPointsGenerater._generate_clustering_based(data, key_rate, cluster_inner_random,
                                                                                  prob, min_num)
        return key_data, key_indices

    @staticmethod
    def _generate_randomly(data: np.ndarray, key_rate: float, prob=None, min_num=0):
        n_samples = data.shape[0]
        indices = np.arange(0, n_samples, 1)
        key_data_num = max(int(n_samples * key_rate), min_num)
        # TODO: choice这个接口有问题，返回的不是唯一值
        key_indices = np.random.choice(indices, key_data_num, p=prob / np.sum(prob), replace=False)
        return data[key_indices], key_indices

    @staticmethod
    def _generate_clustering_based(data: np.ndarray, key_rate: float, is_random=True, prob=None, min_num=0):
        n_samples = data.shape[0]
        cluster_num = int(math.sqrt(n_samples))
        km = KMeans(n_clusters=cluster_num)
        km.fit(data)
        labels = km.labels_
        key_indices = []
        key_data_num = max(int(n_samples * key_rate), min_num)

        if is_random:   # 聚类内部随机采样
            for item in np.unique(labels):
                cur_indices = np.where(labels == item)[0]
                cur_key_num = int(len(cur_indices) * key_data_num / n_samples)
                cur_prob = None
                if prob is not None:
                    cur_prob = prob[cur_indices]
                    cur_prob /= np.sum(cur_prob)
                key_indices.extend(np.random.choice(cur_indices, cur_key_num, p=cur_prob, replace=False))
        else:   # 聚类内部根据各点到聚类中心的距离进行采样
            if prob is not None:
                raise RuntimeError("Custom sampling probability conflicts with distance-based sampling!")
            centroids = km.cluster_centers_
            for i, item in enumerate(np.unique(labels)):
                cur_indices = np.where(labels == item)[0]
                cur_dists = cdist(centroids[i], data[cur_indices])
                sorted_indices = np.argsort(cur_dists)
                cur_key_num = int(len(cur_indices) * key_data_num / n_samples)
                sampled_indices = cur_indices[sorted_indices[np.linspace(0, len(cur_indices), cur_key_num, dtype=int)]]
                key_indices.extend(sampled_indices)

        return data[key_indices], key_indices


