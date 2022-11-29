import math

from autograd import grad
import autograd.numpy as np
from autograd import elementwise_grad
from scipy.spatial.distance import cdist

from utils.loss_grads import nce_loss_grad, norm_grad, umap_sim_grad
from utils.umap_utils import convert_distance_to_probability, find_ab_params


def nce_loss_single_ag(center_embedding, neighbor_embeddings, neg_embeddings, a, b, t=0.15):
    # 为什么不能将这部分计算抽离出去。是因为每次嵌入更新后都需要利用该公式再次计算，所以计算不是重复的。
    center_embedding = center_embedding[np.newaxis, :]
    neighbor_num = neighbor_embeddings.shape[0]

    # 1*(k+m)
    dist = norm_cal(center_embedding, np.concatenate([neighbor_embeddings, neg_embeddings], axis=0))[np.newaxis, :]
    # 1*(k+m)
    umap_sims = convert_distance_to_probability(dist, a, b)

    pos_umap_sims = umap_sims[:, :neighbor_num]
    neg_umap_sims = umap_sims[:, neighbor_num:]
    # k*(1+m)
    total_umap_sims = np.concatenate([pos_umap_sims.T, np.repeat(neg_umap_sims, neighbor_embeddings.shape[0], 0)],
                                     axis=1) / t
    total_umap_sims = np.exp(total_umap_sims)
    # k*1, total_neg_umap_sims也可以直接使用模型训练时的平均值
    pos_nce_sims = (total_umap_sims[:, 0]) / np.sum(total_umap_sims, axis=1)

    loss = np.mean(-np.log(pos_nce_sims))
    return loss


def norm_cal(x1, x2):
    y = (np.sum((x1 - x2)**2, axis=-1))**0.5
    return y


def to_umap(cur_dist):
    return 1.0 / (1.0 + a * cur_dist ** (2 * b))


def mae_loss(center_embedding, neighbors_embeddings, high_sims):
    cur_dist = norm_cal(center_embedding, neighbors_embeddings)

    low_sims = to_umap(cur_dist)
    loss = abs(low_sims - high_sims)
    return loss


nce_grads = elementwise_grad(nce_loss_single_ag)


def my_nce_grads(center_embedding, neighbor_embeddings, neg_embeddings, a, b, t=0.15):
    dx = nce_grads(center_embedding, neighbor_embeddings, neg_embeddings, a, b, t)
    return dx


def nce_grads2sims(sims, t):
    t_sims = sims / t
    normed_sims = t_sims / np.sum(t_sims)
    pos_grads = -1 / t * normed_sims[:, 1:]
    neg_grads = normed_sims[:, 1:] / t
    # print(pos_grads)
    # print("===")
    # print(neg_grads)
    return pos_grads + neg_grads


def nce_loss2sims(sims, t):
    t_sims = sims / t
    normed_sims = t_sims / np.sum(t_sims)
    loss = np.mean(-np.log(normed_sims[1:]))
    return loss

import numpy


if __name__ == '__main__':
    x1 = np.array([-0.21837, 0.32073])
    neighbor_e = np.array(numpy.random.random((1, 2)))
    neg_e = np.array(numpy.random.random((1, 2)))
    x2 = np.array([1.61784, 0.16449])
    y = 0.9458091
    a, b = find_ab_params(1.0, 0.1)

    nce_grads = elementwise_grad(nce_loss_single_ag)

    dx = nce_grads(x1, neighbor_e, neg_e, a, b, 0.15)
    print(dx)
    my_dx = nce_loss_grad(x1, neighbor_e, neg_e, a, b, 0.15)
    print(my_dx)

    dist = norm_cal(x1, np.concatenate([neighbor_e, neg_e], axis=0))[np.newaxis, :]
    # 1*(k+m)
    umap_sims = convert_distance_to_probability(dist, a, b)

    sim_dx = elementwise_grad(nce_loss2sims)(umap_sims, 0.15)
    print("sim dx", sim_dx)
    my_sim_dx = nce_grads2sims(umap_sims, 0.15)
    print(my_sim_dx)

    norm_grads_func = elementwise_grad(norm_cal)
    dx_norm = norm_grads_func(x1, neighbor_e)
    print("dx norm", dx_norm)
    my_dx_norm = norm_grad(x1, neighbor_e, dist[:, :1].T)
    print(np.sum(my_dx_norm, axis=0))

    umap_grads_func = elementwise_grad(to_umap)
    dx_umap = umap_grads_func(dist)
    print("dx_umap", dx_umap)
    my_dx_umap = umap_sim_grad(dist, a, b)
    print(my_dx_umap)