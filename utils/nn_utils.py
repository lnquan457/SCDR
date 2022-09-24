import os
import time

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


class StreamingKNNSearchApprox:
    def __init__(self, beta=10):
        self.beta = beta
        self.searcher = None
        self.pre_embeddings_container = None

    def search(self, k, pre_embeddings, pre_data, query_embeddings, query_data, model_update=False):
        # 这里的embeddings每次都需要保持是最新的
        query_embeddings_container = self._prepare_searcher_data(pre_data, pre_embeddings, query_data,
                                                                 query_embeddings, model_update)

        """
            又获得一个新奇的知识点啊！！！数据的类型不同竟然也会对性能造成这么大的影响（x10+）
            当是nd.array的时候最慢，慢10多倍
            当类型是list的时候，快一些，但是还是慢了2倍多
            当类型是PointContainer，与建树的数据保持一致的时候，取得了最佳的性能
            是跟原型查找链相关的嘛？
        """
        new_k = self.beta * k

        cur_approx_indices = np.empty(new_k)
        cur_other_data = np.empty(shape=(new_k, pre_data.shape[1]))

        nn_indices, nn_dists = self._querying(cur_approx_indices, cur_other_data, k, new_k, query_data,
                                              query_embeddings_container)

        return nn_indices, nn_dists

    def _querying(self, cur_approx_indices, cur_other_data, k, new_k, query_data, query_embeddings_container):
        query_num = len(query_embeddings_container)
        nn_indices = np.zeros((query_num, k), dtype=int)
        nn_dists = np.zeros((query_num, k), dtype=float)
        for i, (e, d) in enumerate(zip(query_embeddings_container, query_data)):
            # TODO：这里也很耗时。有点奇怪??? 是目前主要的性能瓶颈
            res = self.searcher.get_knn(e, new_k)

            for j, item in enumerate(res):
                cur_approx_indices[j] = item[1].label
                cur_other_data[j] = item[1].other_data

            # 这一步是很快的，计算这个数据到所有unfitted数据的距离（这是因为没有fit的数据的嵌入不准确，直接计算距离比较可靠一些）
            dists = cdist(np.expand_dims(d, axis=0), cur_other_data)
            sorted_indices = np.argsort(dists)[0][1:k+1]
            nn_indices[i] = cur_approx_indices[sorted_indices]
            nn_dists[i] = dists[0][sorted_indices]

        return nn_indices, nn_dists

    def _prepare_searcher_data(self, pre_data, pre_embeddings, query_data, query_embeddings, model_update, *args):
        pre_num, dim = pre_embeddings.shape
        query_embeddings_container = add_index_label(query_embeddings.tolist(), pre=pre_num, others=query_data)
        if model_update:
            # 这一步很耗时，但是其实只有在投影函数发生变化之后才需要进行更新，否则只需要追加即可
            pre_embedding_container = add_index_label(pre_embeddings.tolist(), others=pre_data)
            self.pre_embeddings_container = pre_embedding_container + query_embeddings_container
            # ！！！这一步会改变embeddings的顺序
            self.searcher = KDTree(self.pre_embeddings_container, dim)
        else:
            self.pre_embeddings_container = self.pre_embeddings_container + query_embeddings_container
            self.searcher.add_points(query_embeddings_container)
        return query_embeddings_container


# 近似的准则是，从模型已经拟合过的数据的嵌入中选择 beta * k个作为候选，然后准确计算每个待查询数据到其他所有未拟合的数据的距离，然后从这所有数据
# 当中选择最近的k个点。
# TODO：第二个阶段还需要好好思考一下，如何达到效率和精度的最优。
class StreamingKNNSearchApprox2(StreamingKNNSearchApprox):
    def __init__(self, beta=10):
        StreamingKNNSearchApprox.__init__(self, beta)
        self.beta = beta
        self.fitted_embeddings_container = None

    def search(self, k, fitted_embeddings, fitted_data, query_embeddings, query_data, unfitted_data=None):
        fitted_num = self._prepare_searcher_data(fitted_data, fitted_embeddings, unfitted_data)

        new_k = self.beta * k
        unfitted_data = query_data if unfitted_data is None else np.concatenate([unfitted_data, query_data], axis=0)

        unfitted_data_num, dim = unfitted_data.shape
        total_num = new_k + unfitted_data_num
        cur_approx_indices = np.zeros(shape=total_num)
        cur_other_data = np.zeros(shape=(total_num, dim))
        # 因为没有fit的数据的嵌入不准确，这里是为了方便后续直接计算新数据与其他unfit的数据的距离，比较可靠一些
        cur_approx_indices[-unfitted_data_num:] = np.arange(fitted_num, fitted_num + unfitted_data_num, 1)
        cur_other_data[-unfitted_data_num:] = unfitted_data

        return self._querying(cur_approx_indices, cur_other_data, k, new_k, query_data, query_embeddings.tolist())

    def _prepare_searcher_data(self, fitted_data, fitted_embeddings, unfitted_data, *args):
        fitted_num, dim = fitted_embeddings.shape
        if unfitted_data is None:
            # 模型更新了之后，之前数据的嵌入结果也需要更新，所以要重新构建
            # 这一步很耗时，但是其实只有在投影函数发生变化之后才需要进行更新，否则只需要追加即可
            self.fitted_embeddings_container = add_index_label(fitted_embeddings.tolist(), others=fitted_data)
            # ！！！这一步会改变embeddings的顺序
            self.searcher = KDTree(self.fitted_embeddings_container, dim)
        return fitted_num
