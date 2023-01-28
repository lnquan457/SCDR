import os
import time

import numba
import numpy as np
from pynndescent import NNDescent
from scipy.spatial.ckdtree import cKDTree
from scipy.spatial.distance import cdist
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


def compute_accurate_knn(flattened_data, k, neighbors_cache_path=None, pairwise_cache_path=None, metric="euclidean",
                         include_self=False):
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
        if include_self:
            knn_indices = sorted_indices[:, :k]
        else:
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
                      metric="euclidean", max_candidates=60, accelerate=False, include_self=False):
    flattened_data = all_data.reshape((len(all_data), np.product(all_data.shape[1:])))
    # 精确的KNN，比较慢
    if not accelerate:
        knn_indices, knn_distances = compute_accurate_knn(flattened_data, k, neighbors_cache_path, pairwise_cache_path,
                                                          include_self=include_self)
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
        if self.method == ANNOY:
            self._build_annoy_index(initial_data)
        elif self.method == KD_TREE:
            dim = len(initial_data[0])
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

    def search(self, data, k, query=True, adding=True):
        sta = time.time()
        if self.method == KD_TREE:
            data = add_index_label(data.tolist())

        if adding:  # 建树或者添加节点
            if self.searcher is None:
                self._init_searcher(data)
            else:
                if self.method == ANNOY:
                    self._build_annoy_index(data)
                elif self.method == KD_TREE:
                    self.searcher.add_points(data)

        if not query:
            return None, None

        assert self.searcher is not None

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
                res = self.searcher.get_knn(data[i], k + 1, return_dist_sq=True)
                nn_indices[i] = [item[1].label for item in res][1:]
                nn_dists[i] = [item[0] for item in res][1:]
        # print("Search cost time:", time.time() - sta)
        return nn_indices, nn_dists


# 近似的准则是，从模型已经拟合过的数据的嵌入中选择 beta * k个作为候选，然后准确计算每个待查询数据到其他所有未拟合的数据的距离，然后从这所有数据
# 当中选择最近的k个点。
# TODO：第二个阶段还需要好好思考一下，如何达到效率和精度的最优。
class StreamingANNSearchKD:
    def __init__(self, beta=10):
        self.searcher = None
        self.beta = beta
        self.fitted_embeddings_container = None

    def _querying(self, cur_approx_indices, cur_other_data, k, new_k, pre_data, query_data, query_embeddings_container):
        # TODO：这里也很耗时。有点奇怪??? 是目前主要的性能瓶颈
        res = self.searcher.get_knn(query_embeddings_container, new_k)
        cur_approx_indices[:new_k] = np.array(res)[:, 2].astype(int)
        cur_other_data[:new_k] = pre_data[cur_approx_indices[:new_k]]

        # 这一步是很快的，计算这个数据到所有unfitted数据的距离（这是因为没有fit的数据的嵌入不准确，直接计算距离比较可靠一些）
        dists = cdist(query_data, cur_other_data).squeeze()
        sorted_indices = np.argsort(dists)[:k]
        nn_indices = cur_approx_indices[sorted_indices]
        nn_dists = dists[sorted_indices]
        return nn_indices, nn_dists

    def search(self, k, pre_embeddings, pre_data, fitted_num, query_embeddings, query_data, update=False):
        self._prepare_searcher_data(pre_data[:fitted_num], pre_embeddings[:fitted_num], update)

        new_k = self.beta * k
        unfitted_data_num = pre_data.shape[0] - fitted_num

        total_num = new_k + unfitted_data_num
        cur_approx_indices = np.zeros(shape=total_num, dtype=int)
        cur_other_data = np.zeros(shape=(total_num, pre_data.shape[1]))
        # 因为没有fit的数据的嵌入不准确，这里是为了方便后续直接计算新数据与其他unfit的数据的距离，比较可靠一些
        if unfitted_data_num > 0:
            cur_approx_indices[-unfitted_data_num:] = np.arange(fitted_num, fitted_num + unfitted_data_num, 1)
            unfitted_data = pre_data[fitted_num:]
            cur_other_data[-unfitted_data_num:] = unfitted_data

        return self._querying(cur_approx_indices, cur_other_data, k, new_k, pre_data, query_data,
                              query_embeddings.squeeze().tolist())

    def _prepare_searcher_data(self, fitted_data, fitted_embeddings, update, *args):
        fitted_num, dim = fitted_embeddings.shape
        if update or self.searcher is None:
            # TODO: 每次都需要重新构建container，并且重新建树，导致非常耗时。
            self.fitted_embeddings_container = add_index_label(fitted_embeddings.tolist(), others=fitted_data)
            # ！！！这一步会改变embeddings的顺序
            self.searcher = KDTree(self.fitted_embeddings_container, dim)
        return fitted_num


class StreamingANNSearchAnnoy:
    def __init__(self, beta=10, update_iter=500, automatic_beta=True):
        self._searcher = None
        self._beta = beta
        self._update_iter = update_iter
        self._fitted_num = 0
        self._opt_embedding_indices = np.array([], dtype=int)
        self._optimized_data = None
        self._infer_embedding_indices = np.array([], dtype=int)
        self._inferred_data = None
        self._automatic_beta = automatic_beta

    def search_2(self, k, pre_embeddings, pre_data, query_embeddings, query_data, unfitted_num, update=False):
        if update:
            self._build_annoy_index(pre_embeddings[-unfitted_num:])
        elif (pre_embeddings.shape[0] - self._fitted_num) >= self._update_iter:
            self._build_annoy_index(pre_embeddings)

        if not self._automatic_beta:
            new_k = self._beta * k
        else:
            new_k = int(0.15 * np.sqrt(pre_data.shape[0]) * k)

        query_num = query_data.shape[0]
        if query_num == 1:
            candidate_indices = self._searcher.get_nns_by_vector(query_embeddings.squeeze(), new_k)
            candidate_indices = np.array(candidate_indices, dtype=int)
            if not update:
                candidate_indices = np.union1d(candidate_indices,
                                               np.arange(self._fitted_num, pre_data.shape[0]).astype(int))

            candidate_data = pre_data[candidate_indices]

            dists = cdist(query_data, candidate_data).squeeze()
            sorted_indices = np.argsort(dists)[:k].astype(int)
            final_indices = candidate_indices[sorted_indices][np.newaxis, :]
            final_dists = dists[sorted_indices][np.newaxis, :]
            candidate_indices = [candidate_indices]
            dists = [dists]
        else:
            # ====================================for batch process=====================================
            final_indices = np.empty((query_num, k), dtype=int)
            final_dists = np.empty((query_num, k), dtype=float)
            final_candidate_indices = []
            dists = []
            unfitted_data_indices = np.arange(self._fitted_num, pre_data.shape[0]).astype(int)
            for i in range(query_num):
                candidate_indices = self._searcher.get_nns_by_vector(query_embeddings[i], new_k)
                candidate_indices = np.array(candidate_indices, dtype=int)

                if not update:
                    candidate_indices = np.union1d(candidate_indices, unfitted_data_indices)

                final_candidate_indices.append(candidate_indices)
                cur_dists = cdist(query_data[i][np.newaxis, :], pre_data[candidate_indices]).squeeze()
                dists.append(cur_dists)
                sorted_indices = np.argsort(cur_dists)[:k]
                final_indices[i] = candidate_indices[sorted_indices]
                final_dists[i] = cur_dists[sorted_indices]

            candidate_indices = final_candidate_indices
            # ====================================for batch process=====================================

        return final_indices, final_dists, candidate_indices, dists

    def search(self, k, pre_embeddings, pre_data, query_embeddings, query_data, optimized, update=False):
        update = update or (pre_embeddings.shape[0] - self._fitted_num) % self._update_iter == 0
        if update:
            self._build_annoy_index(pre_embeddings)

        if not self._automatic_beta:
            new_k = self._beta * k
        else:
            new_k = 0.2 * np.sqrt(pre_embeddings.shape[0]) * k

        candidate_indices, knn_dists = self._searcher.get_nns_by_vector(query_embeddings.squeeze(), new_k,
                                                                        include_distances=True)
        candidate_indices = np.array(candidate_indices, dtype=int)
        candidate_data = pre_data[candidate_indices]

        if not update:
            pre_last_data = pre_data[-1][np.newaxis, :]
            if optimized:
                self._opt_embedding_indices = np.append(self._opt_embedding_indices, pre_embeddings.shape[0] - 1)
                self._optimized_data = pre_last_data if self._optimized_data is None else \
                    np.concatenate([self._optimized_data, pre_last_data], axis=0)
            else:
                self._infer_embedding_indices = np.append(self._infer_embedding_indices, pre_embeddings.shape[0] - 1)
                self._inferred_data = pre_last_data if self._inferred_data is None else \
                    np.concatenate([self._inferred_data, pre_last_data], axis=0)

            if len(self._opt_embedding_indices) > 0:
                candidate_indices = np.concatenate([candidate_indices, self._opt_embedding_indices])
                candidate_data = np.concatenate([candidate_data, self._optimized_data], axis=0)

            if len(self._infer_embedding_indices) > 0:
                dists = cdist(query_embeddings, pre_embeddings[self._infer_embedding_indices])
                selected_indices = np.where(dists < np.max(knn_dists))[0]
                if len(selected_indices) > 0:
                    candidate_indices = np.concatenate(
                        [candidate_indices, self._infer_embedding_indices[selected_indices]])
                    candidate_data = np.concatenate([candidate_data, self._inferred_data[selected_indices]], axis=0)

        dists = cdist(query_data, candidate_data).squeeze()
        sorted_indices = np.argsort(dists)[:k].astype(int)
        final_indices = candidate_indices[sorted_indices]
        final_dists = dists[sorted_indices]

        return final_indices, final_dists

    def _build_annoy_index(self, embeddings):
        if self._searcher is None:
            self._searcher = AnnoyIndex(embeddings.shape[1], 'euclidean')
        else:
            self._searcher.unbuild()

        for i in range(embeddings.shape[0]):
            self._searcher.add_item(i, embeddings[i])

        self._searcher.build(10)
        self._fitted_num = embeddings.shape[0]
        self._opt_embedding_indices = np.array([], dtype=int)
        self._optimized_data = None
        self._infer_embedding_indices = np.array([], dtype=int)
        self._inferred_data = None


# @numba.jit(nopython=True)
def heapK(ary, nums, k):
    if nums <= k:
        return ary

    ks = ary[:k]
    build_heap(ks, k)  # 构建大顶堆（先不排序）

    for index in range(k, nums):
        ele = ary[index]
        if ks[0] > ele:
            ks[0] = ele
            downAdjust(ks, 0, k)

    return ks


# @numba.jit(nopython=True)
def build_heap(ary_list, k):
    index = k // 2 - 1  # 最后一个非叶子结点
    while index >= 0:
        downAdjust(ary_list, index, k)
        index -= 1


# @numba.jit(nopython=True)
def downAdjust(ary_list, parent_index, k):
    tmp = ary_list[parent_index]
    child_index = 2 * parent_index + 1

    while child_index < k:
        if child_index + 1 < k and ary_list[child_index + 1] > ary_list[child_index]:
            child_index += 1

        if tmp >= ary_list[child_index]:
            break

        ary_list[parent_index] = ary_list[child_index]
        parent_index = child_index
        child_index = 2 * parent_index + 1

    ary_list[parent_index] = tmp


def find_k_minimums(data, k):
    nums = data.shape[0]
    return heapK(data, nums, k)
