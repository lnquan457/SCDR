import os
import time

import numpy as np
from pynndescent import NNDescent
from sklearn.metrics import pairwise_distances
from annoy import AnnoyIndex

from utils.kd_tree import KDTree, add_index_label
from utils.logger import InfoLogger


def cal_snn_similarity(knn, cache_path=None):
    """
    基于KNN图计算SNN相似度
    :param cache_path:
    :param knn:
    :return:
    """
    if cache_path is not None and os.path.exists(cache_path):
        _, snn_sim = np.load(cache_path)
        InfoLogger.info("directly load accurate neighbor_graph from {}".format(cache_path))
        return knn, snn_sim

    snn_sim = np.zeros_like(knn)
    n_samples, n_neighbors = knn.shape
    for i in range(n_samples):
        sample_nn = knn[i]
        for j, neighbor_idx in enumerate(sample_nn):
            neighbor_nn = knn[int(neighbor_idx)]
            snn_num = len(np.intersect1d(sample_nn, neighbor_nn))
            snn_sim[i][j] = snn_num / n_neighbors
    if cache_path is not None and not os.path.exists(cache_path):
        np.save(cache_path, [knn, snn_sim])
        InfoLogger.info("successfully compute snn similarity and save to {}".format(cache_path))
    return knn, snn_sim


def compute_accurate_knn(flattened_data, k, neighbors_cache_path=None, pairwise_cache_path=None, metric="euclidean"):
    cur_path = None
    if neighbors_cache_path is not None:
        cur_path = neighbors_cache_path.replace(".npy", "_ac.npy")

        # cur_path = neighbors_cache_path.replace(".npy", "_opt.npy")
        # cur_path = neighbors_cache_path.replace(".npy", "_cls.npy")

    if cur_path is not None and os.path.exists(cur_path):
        knn_indices, knn_distances = np.load(cur_path)
        InfoLogger.info("directly load accurate neighbor_graph from {}".format(cur_path))
    else:
        preload = flattened_data.shape[0] <= 30000

        pairwise_distance = get_pairwise_distance(flattened_data, metric, pairwise_cache_path, preload=preload)
        sorted_indices = np.argsort(pairwise_distance, axis=1)
        knn_indices = sorted_indices[:, 1:k + 1]
        knn_distances = []
        for i in range(knn_indices.shape[0]):
            knn_distances.append(pairwise_distance[i, knn_indices[i]])
        knn_distances = np.array(knn_distances)
        if cur_path is not None:
            np.save(cur_path, [knn_indices, knn_distances])
            InfoLogger.info("successfully compute accurate neighbor_graph and save to {}".format(cur_path))
    return knn_indices, knn_distances


def compute_knn_graph(all_data, neighbors_cache_path, k, pairwise_cache_path,
                      metric="euclidean", max_candidates=60, accelerate=False):
    flattened_data = all_data.reshape((len(all_data), np.product(all_data.shape[1:])))
    # 精确的KNN，比较慢
    if not accelerate:
        knn_indices, knn_distances = compute_accurate_knn(flattened_data, k, neighbors_cache_path, pairwise_cache_path)
        return knn_indices, knn_distances

    # 近似的KNN，比较快
    if neighbors_cache_path is not None and os.path.exists(neighbors_cache_path):
        neighbor_graph = np.load(neighbors_cache_path)
        knn_indices, knn_distances = neighbor_graph
        InfoLogger.info("directly load approximate neighbor_graph from {}".format(neighbors_cache_path))
    else:
        # number of trees in random projection forest
        n_trees = 5 + int(round((all_data.shape[0]) ** 0.5 / 20.0))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(all_data.shape[0]))))

        nnd = NNDescent(
            flattened_data,
            n_neighbors=k + 1,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=max_candidates,
            verbose=False
        )
        # 获取近邻点下标和距离
        knn_indices, knn_distances = nnd.neighbor_graph
        knn_indices = knn_indices[:, 1:]
        knn_distances = knn_distances[:, 1:]
        # 缓存邻近图
        if neighbors_cache_path is not None:
            np.save(neighbors_cache_path, [knn_indices, knn_distances])
        InfoLogger.info("successfully compute approximate neighbor_graph and save to {}".format(neighbors_cache_path))
    return knn_indices, knn_distances


def get_pairwise_distance(flattened_data, metric="euclidean", pairwise_distance_cache_path=None, preload=False):
    if pairwise_distance_cache_path is not None and preload and os.path.exists(pairwise_distance_cache_path):
        pairwise_distance = np.load(pairwise_distance_cache_path)
        InfoLogger.info("directly load pairwise distance from {}".format(pairwise_distance_cache_path))
    else:
        # InfoLogger.info("computing pairwise distance...")
        pairwise_distance = pairwise_distances(flattened_data, metric=metric, squared=False)
        # pairwise_distance = pdist(flattened_data, metric="sqeuclidean")
        pairwise_distance[pairwise_distance < 1e-12] = 0.0
        # pairwise_distance = np.divide(pairwise_distance, flattened_data.shape[1])
        if preload and pairwise_distance_cache_path is not None:
            np.save(pairwise_distance_cache_path, pairwise_distance)
            InfoLogger.info(
                "successfully compute pairwise distance and save to {}".format(pairwise_distance_cache_path))
    return pairwise_distance


ANNOY = "annoy"
KD_TREE = "kd_tree"


class StreamingKNNSearcher:
    def __init__(self, method):
        self.searcher = None
        self.method = method

    def _init_searcher(self, initial_data):
        dim = len(initial_data[0])
        if self.method == ANNOY:
            self._build_annoy_index(initial_data)
        elif self.method == KD_TREE:
            self.searcher = KDTree(initial_data, dim)
        else:
            raise RuntimeError("Unsupported knn search method! Please ensure the 'method' is one of annoy/kd_tree")

    def _build_annoy_index(self, data):
        if self.searcher is None:
            pre_nums = 0
            self.searcher = AnnoyIndex(data.shape[1], 'euclidean')
        else:
            pre_nums = self.searcher.get_n_items()
            self.searcher.unbuild()

        for i in range(data.shape[0]):
            self.searcher.add_item(i + pre_nums, data[i])
        self.searcher.build(100)

    def search(self, data, k, just_add_new_data=False):
        sta = time.time()
        if self.method == KD_TREE:
            data = add_index_label(data.tolist())
        # 建树或者添加节点
        if self.searcher is None:
            self._init_searcher(data)
        else:
            if self.method == ANNOY:
                self._build_annoy_index(data)
            elif self.method == KD_TREE:
                self.searcher.add_points(data)

        if just_add_new_data:
            return None, None

        # 查询k近邻
        data_num = len(data)
        nn_indices = np.empty((data_num, k), dtype=int)
        nn_dists = np.empty((data_num, k), dtype=float)

        if self.method == ANNOY:
            for i in range(data_num):
                cur_indices, cur_dists = self.searcher.get_nns_by_vector(data[i], k + 1, include_distances=True)
                nn_indices[i], nn_dists[i] = cur_indices[1:], cur_dists[1:]
        elif self.method == KD_TREE:
            for i in range(data_num):
                res = self.searcher.get_knn(data[i], k, return_dist_sq=True)
                nn_indices[i] = [item[1].label for item in res]
                nn_dists[i] = [item[0] for item in res]
        print("Search cost time:", time.time() - sta)
        return nn_indices, nn_dists

