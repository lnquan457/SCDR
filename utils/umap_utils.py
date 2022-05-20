#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy
import numpy as np
import torch
import scipy
from scipy.sparse import coo_matrix, csr_matrix
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from utils.nn_utils import cal_snn_similarity, compute_accurate_knn

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf


def get_graph_elements(graph_, n_epochs):
    """
    gets elements of graphs, weights, and number of epochs per edge

    Parameters
    ----------
    graph_ : scipy.sparse.csr.csr_matrix
        umap graph of probabilities
    n_epochs : int
        maximum number of epochs per edge

    Returns
    -------
    graph scipy.sparse.csr.csr_matrix
        umap graph
    epochs_per_sample np.array
        number of epochs to train each sample for
    head np.array
        edge head
    tail np.array
        edge tail
    weight np.array
        edge weight
    n_vertices int
        number of verticies in graph
    """
    ### should we remove redundancies () here??
    # graph_ = remove_redundant_edges(graph_)

    # CSR -> COO
    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    # 主要是为了方便之后的采样策略，对于权重非常小的边，可以看作两点之间没有连通性，消除即可
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    # 删除0值的键值对
    graph.eliminate_zeros()
    # get epochs per sample based upon edge probability
    # 每个epoch中每条边被采样的次数，根据权重求出来的，权重越大采样次数越多
    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    # 通过边来构造起点和终点
    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices


def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights of how much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    # weights = np.ones_like(weights) / 2
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result


def construct_edge_dataset(
        X,
        graph_,
        n_epochs
):
    """
    Construct a tf.data.Dataset of edges, sampled by edge weight.
    """

    # get data from graph
    # 这里的graph是去除了重复点对关系之后和一些权重非常小的边之后的权重图
    # head和tail就分别表示了各数据点对的开始和结尾
    graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
        graph_, n_epochs
    )

    # 根据每个点在一个epoch中的采样次数进行重复，此时的数据集就是根据权重采样得到的了
    # 权重越大表示越可能存在边，因此被采样的次数也越多
    edges_to_exp, edges_from_exp = (
        np.repeat(head, epochs_per_sample.astype("int")),
        np.repeat(tail, epochs_per_sample.astype("int")),
    )

    # shuffle edges
    # 生成一个随机序列
    shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
    # 这就相当于对数据进行了打乱
    edges_to_exp = edges_to_exp[shuffle_mask]
    edges_from_exp = edges_from_exp[shuffle_mask]

    embedding_to_from_indices = np.array([edges_to_exp, edges_from_exp])
    # 这里又对所有的边进行了加倍
    embedding_to_from_indices_re = np.repeat(embedding_to_from_indices, 2, 1)
    np.random.shuffle(embedding_to_from_indices_re)
    embedding_to_from_data = X[embedding_to_from_indices[0, :]], X[embedding_to_from_indices[1, :]]

    return embedding_to_from_data, len(edges_to_exp), weight

    # # create edge iterator
    #     edge_dataset = tf.data.Dataset.from_tensor_slices(
    #         (edges_to_exp, edges_from_exp)
    #     )
    #     edge_dataset = edge_dataset.repeat()
    #     edge_dataset = edge_dataset.shuffle(10000)
    #     edge_dataset = edge_dataset.map(
    #         gather_X, num_parallel_calls=tf.data.experimental.AUTOTUNE
    #     )
    #     edge_dataset = edge_dataset.batch(batch_size, drop_remainder=True)
    #     edge_dataset = edge_dataset.prefetch(10)
    # return edge_dataset, batch_size, len(edges_to_exp), head, tail, weight


def fuzzy_simplicial_set_partial(all_knn_indices, all_knn_dists, all_raw_knn_weights, update_indices,
                                 set_op_mix_ratio=1.0, local_connectivity=1.0,
                                 apply_set_operations=True, symmetric="TSNE", return_dists=None):
    updated_knn_indices = all_knn_indices[update_indices]
    updated_knn_dists = all_knn_dists[update_indices].astype(np.float32)
    n_neighbors = all_knn_indices.shape[1]

    sigmas, rhos = smooth_knn_dist(updated_knn_dists, float(n_neighbors), local_connectivity=float(local_connectivity))

    rows, cols, vals, dists = compute_membership_strengths(
        updated_knn_indices, updated_knn_dists, sigmas, rhos, return_dists
    )

    cur_updated_knn_weights = vals.reshape(updated_knn_indices.shape)
    all_raw_knn_weights[update_indices] = cur_updated_knn_weights

    total_n_samples = all_knn_indices.shape[0]
    new_rows = np.ravel(np.repeat(np.expand_dims(np.arange(0, total_n_samples, 1), axis=1), axis=1, repeats=n_neighbors))
    new_cols = np.ravel(all_knn_indices)
    new_vals = np.ravel(all_raw_knn_weights)

    result = scipy.sparse.coo_matrix(
        (new_vals, (new_rows, new_cols)), shape=(total_n_samples, total_n_samples)
    )
    result.eliminate_zeros()

    if apply_set_operations:
        result = apply_set(all_raw_knn_weights, result, set_op_mix_ratio, symmetric)
    result.eliminate_zeros()

    return result, sigmas, rhos, all_raw_knn_weights


def apply_set(origin_knn_weights, result, set_op_mix_ratio, symmetric):
    transpose = result.transpose()
    # umap的对称化
    if symmetric == "UMAP":
        prod_matrix = result.multiply(transpose)
        result = (
                set_op_mix_ratio * (result + transpose - prod_matrix)
                + (1.0 - set_op_mix_ratio) * prod_matrix
        )
    elif symmetric == "TSNE":
        # tsne的对称化
        result = (result + transpose) / 2
    elif symmetric == "CUSTOM":
        # 自定义对称化
        result_arr = result.A
        extra_neighbor = np.expand_dims(np.min(origin_knn_weights, axis=1) - 0.0001, axis=1) * result_arr.T
        symm_result = result_arr + extra_neighbor
        result = scipy.sparse.csr_matrix(symm_result)
    return result


def fuzzy_simplicial_set(
    X,
    n_neighbors,
    knn_indices=None,
    knn_dists=None,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    apply_set_operations=True,
    return_dists=None,
    symmetric="UMAP",
    labels=None
):
    # 如果没有邻近图信息则在这里进行计算
    if knn_indices is None or knn_dists is None:
        pass

    knn_dists = knn_dists.astype(np.float32)

    # ===========================直接进行截断，只取前50%个邻居进行优化=======================
    # n_neighbors = int(n_neighbors * 0.5)
    # knn_indices = knn_indices[:, :n_neighbors]
    # knn_dists = knn_dists[:, :n_neighbors]
    # ====================================================================================

    # 将邻近点的距离由离散的转换为连续的，sigma表示用于进行正则化的因子，代表局部的黎曼度量
    # rho表示每个点到最近邻居的距离
    sigmas, rhos = smooth_knn_dist(
        knn_dists,
        float(n_neighbors),
        local_connectivity=float(local_connectivity),
    )

    # 根据sigma和rho来计算图的权重信息，利用rows和cols来表示这个权重图，vals就是权重，都是[n_samples * k, 1]
    rows, cols, vals, dists = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos, return_dists
    )

    # knn_sim = np.reshape(vals, knn_indices.shape)
    # knn_sim = knn_sim / np.expand_dims(np.sum(knn_sim, axis=1), axis=1)
    # snn_sim = cal_snn_similarity(knn_indices)[1]
    # snn_sim = snn_sim / np.expand_dims(np.sum(snn_sim, axis=1), axis=1)
    #
    # alpha = 0
    # vals = alpha * vals + (1 - alpha) * snn_sim.flatten()

    origin_knn_weights = vals.reshape(knn_indices.shape)

    # 转换为稀疏矩阵，由col和data构成，都是[n_samples * k, 1]，其中col（n = i*k+j）表示邻居点关系，data就是对应的权重
    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    # 消除0值
    result.eliminate_zeros()

    # 将多个模糊单纯形集合并
    if apply_set_operations:
        transpose = result.transpose()

        # transpose = np.array(transpose.todense())
        # transpose[5009] *= 0
        # transpose = csr_matrix(transpose).tocoo()
        # 这里把COO形式存储的result转换为了CSR形式

        # umap的对称化
        if symmetric == "UMAP":
            prod_matrix = result.multiply(transpose)
            result = (
                set_op_mix_ratio * (result + transpose - prod_matrix)
                + (1.0 - set_op_mix_ratio) * prod_matrix
            )
        elif symmetric == "TSNE":
            # tsne的对称化
            result = (result + transpose) / 2
        elif symmetric == "CUSTOM":
            # 自定义对称化
            result_arr = result.A
            extra_neighbor = np.expand_dims(np.min(origin_knn_weights, axis=1) - 0.0001, axis=1) * result_arr.T
            symm_result = result_arr + extra_neighbor
            result = scipy.sparse.csr_matrix(symm_result)
    result.eliminate_zeros()

    if return_dists is None:
        return result, sigmas, rhos, origin_knn_weights
    else:
        if return_dists:
            dmat = coo_matrix(
                (dists, (rows, cols)), shape=(X.shape[0], X.shape[0])
            )

            dists = dmat.maximum(dmat.transpose()).todok()
        else:
            dists = None

        return result, sigmas, rhos, origin_knn_weights, dists


def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """
    Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    # 获取所有点到其最近邻居的距离
    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        # 获得第i个点到其k个邻居的距离
        ith_distances = distances[i]
        # 获得所有距离大于0的点（有连通的）
        non_zero_dists = ith_distances[ith_distances > 0.0]
        # 如果联通点数大于局部连通要求，则直接获取最近距离rho，这个可能是由于距离度量方式不同而导致的
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            # 如果local_connectivity是分数就进行插值
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            """
            遍历当前点的所有邻居节点，与t-sne中根据perplexity查找合适的高斯分布方差类似，
            只不过标准不同，这里要找的是一个正则化因子，在这个因子的约束下，将最近邻居点距离看作是
            一个黎曼度量单位。都是找到一个合适的rho，将其看作局部的黎曼度量来描述这个数据的局部特征
            这一步按道理来说，应该也是效率极低的。
            """
            for j in range(1, distances.shape[1]):
                # d表示以最近距离归一化后的距离
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break
            # 二分查找的方式
            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0
        # 正则化因子sigma
        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho


def compute_membership_strengths(
    knn_indices, knn_dists, sigmas, rhos, return_dists=False
):
    """
    Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)
    if return_dists:
        dists = np.zeros(knn_indices.size, dtype=np.float32)
    else:
        dists = None

    for i in range(n_samples):
        for j in range(n_neighbors):
            # 不一定获取到完整的k个近邻
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            # 到自身的权重为0
            if knn_indices[i, j] == i:
                val = 0.0
            # 最近的邻居点的权重
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                # 根据sigma和rho进行正则化
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]

    return rows, cols, vals, dists


def find_ab_params(spread, min_dist):
    """
    Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """
    # 预先定义好的一个类似于t分布的函数，当a和b都为1的时候就是自由度为1的t分布
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    # 模拟300个从0到3*spread线性增长的点，表示低维空间上的距离
    xv = np.linspace(0, spread * 3, 300)
    # y表示对应的相似度
    yv = np.zeros(xv.shape)
    # 小于指定最小距离的则直接设置相似度为1
    yv[xv < min_dist] = 1.0
    # 其余的按照之前相同的公式 e^(-((dist(i, j) - min_dist) / rho)) 进行设置，但是这里固定了 spread即rho值 为1
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


def convert_distance_to_probability(distances, a=1.0, b=1.0):
    """
     convert distance representation into probability,
        as a function of a, b params
    """
    return 1.0 / (1.0 + a * distances ** (2 * b))


def compute_cross_entropy(probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0):
    """
    Compute cross entropy between low and high probability
    """
    # cross entropy
    attraction_term = -probabilities_graph * torch.log(
        torch.clip(probabilities_distance, EPS, 1.0)
    )
    repellent_term = (
            -(1.0 - probabilities_graph)
            * torch.log(torch.clip(1.0 - probabilities_distance, EPS, 1.0))
            * repulsion_strength
    )

    # balance the expected losses between attraction and repel
    CE = attraction_term + repellent_term
    return attraction_term, repellent_term, CE


def compute_local_membership(knn_dist, knn_indices, local_connectivity=1):
    knn_dist = knn_dist.astype(np.float32)

    # 将邻近点的距离由离散的转换为连续的，sigma表示用于进行正则化的因子，代表局部的黎曼度量
    # rho表示每个点到最近邻居的距离
    sigmas, rhos = smooth_knn_dist(
        knn_dist,
        float(knn_indices.shape[1]),
        local_connectivity=float(local_connectivity),
    )

    # 根据sigma和rho来计算图的权重信息，利用rows和cols来表示这个权重图，vals就是权重，都是[n_samples * k, 1]
    rows, cols, vals, dists = compute_membership_strengths(
        knn_indices+1, knn_dist, sigmas, rhos, False
    )
    # 这里由于所有的邻居点都是采样得到的，那么其实各个局部邻域之间是没有重合的，所以不需要进行全局的单纯形融合
    return vals