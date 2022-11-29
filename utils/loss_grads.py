import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from utils.umap_utils import convert_distance_to_probability


def mae_loss_gard(center_embedding, neighbor_embeddings, high_sims, a, b, sigma=None):
    center_embedding = center_embedding[np.newaxis, :]
    cur_dist = np.linalg.norm(center_embedding - neighbor_embeddings, axis=-1)
    low_sims = convert_distance_to_probability(cur_dist, a, b)
    grad_per = np.ones((len(low_sims), neighbor_embeddings.shape[1]))
    grad_per[low_sims - high_sims < 0] = -1
    grad_per *= umap_sim_grad(cur_dist, a, b)[:, np.newaxis] * norm_grad(center_embedding, neighbor_embeddings,
                                                                         cur_dist[:, np.newaxis])
    return np.mean(grad_per, axis=0)


def mse_loss_grad(center_embedding, neighbor_embeddings, high_sims, a, b, sigma=None):
    center_embedding = center_embedding[np.newaxis, :]
    cur_dist = np.linalg.norm(center_embedding - neighbor_embeddings, axis=-1)
    low_sims = convert_distance_to_probability(cur_dist, a, b)
    grad_per = ((low_sims - high_sims) * umap_sim_grad(cur_dist, a, b))[:, np.newaxis] * \
               norm_grad(center_embedding, neighbor_embeddings, cur_dist[:, np.newaxis])

    # if sigma is not None:
    #     sigma_sim = convert_distance_to_probability(sigma, a, b)
    #     mse_indices = np.where(low_sims >= sigma_sim)[0]
    #     grad_per = grad_per[mse_indices]

    return np.mean(grad_per, axis=0)


def huber_loss_gard(center_embedding, neighbor_embeddings, high_sims, a, b, sigma=None):
    center_embedding = center_embedding[np.newaxis, :]
    cur_dist = np.linalg.norm(center_embedding - neighbor_embeddings, axis=-1)
    low_sims = convert_distance_to_probability(cur_dist, a, b)
    sigma_sim = convert_distance_to_probability(sigma, a, b)
    mse_indices = np.where(low_sims >= sigma_sim)[0]
    mae_indices = np.where(low_sims < sigma_sim)[0]

    grad_per = np.ones((len(low_sims), neighbor_embeddings.shape[1]))
    grad_per[mae_indices][low_sims[mae_indices] - high_sims[mae_indices] <= 0] = -1
    grad_per[mae_indices] *= sigma_sim * umap_sim_grad(cur_dist[mae_indices], a, b)[:, np.newaxis] * \
                             norm_grad(center_embedding, neighbor_embeddings[mae_indices],
                                       cur_dist[mae_indices][:, np.newaxis])

    grad_per[mse_indices] = ((low_sims[mse_indices] - high_sims[mse_indices]) *
                             umap_sim_grad(cur_dist[mse_indices], a, b))[:, np.newaxis] * \
                            norm_grad(center_embedding, neighbor_embeddings[mse_indices],
                                      cur_dist[mse_indices][:, np.newaxis])
    return np.mean(grad_per, axis=0)


def nce_loss_grad(center_embedding, neighbor_embeddings, neg_embeddings, a, b, t=0.15):
    center_embedding = center_embedding[np.newaxis, :]
    neighbor_num = neighbor_embeddings.shape[0]

    # 1*(k+m)
    dist = cdist(center_embedding, np.concatenate([neighbor_embeddings, neg_embeddings], axis=0))
    # 1*(k+m)
    umap_sims = convert_distance_to_probability(dist, a, b)

    pos_dist = dist[:, :neighbor_num]
    neg_dist = dist[:, neighbor_num:]
    pos_umap_sims = umap_sims[:, :neighbor_num]
    neg_umap_sims = umap_sims[:, neighbor_num:]
    # k*(1+m)
    total_umap_sims = np.concatenate([pos_umap_sims.T, np.repeat(neg_umap_sims, neighbor_embeddings.shape[0], 0)],
                                     axis=1) / t
    # k*(1+m), total_neg_umap_sims也可以直接使用模型训练时的平均值
    total_nce_sims = total_umap_sims / np.sum(total_umap_sims, axis=1)[:, np.newaxis]
    pos_grads = -1 / t * (np.sum(total_nce_sims[:, 1:], axis=1) * umap_sim_grad(pos_dist, a, b)).T \
                * norm_grad(center_embedding, neighbor_embeddings, pos_dist.T)
    neg_grads = 1 / t * np.sum((total_nce_sims[:, 1:] * umap_sim_grad(neg_dist, a, b))[:, :, np.newaxis] *
                               norm_grad(center_embedding, neg_embeddings, neg_dist.T)[np.newaxis, :, :], axis=1)

    return np.mean(pos_grads + neg_grads, axis=0)


# @jit
def ce_loss_grad(center_embedding, neighbor_embeddings, high_sims, a, b, sigma=None):
    center_embedding = center_embedding[np.newaxis, :]
    cur_dist = np.linalg.norm(center_embedding - neighbor_embeddings, axis=-1)
    low_sims = convert_distance_to_probability(cur_dist, a, b)

    grad_per = -1 * ((high_sims / low_sims - (1 - high_sims) / (1 - low_sims)) * umap_sim_grad(cur_dist, a, b))[:,
                    np.newaxis] * norm_grad(center_embedding, neighbor_embeddings, cur_dist[:, np.newaxis])
    return np.mean(grad_per, axis=0)


@jit
def umap_sim_grad(dist, a, b):
    return -2 * a * b * np.power((1 + a * np.power(dist, 2 * b)), -2) * np.power(dist, 2 * b - 1)


@jit
def norm_grad(pred, gt, dist):
    return np.power(np.power(dist, 2), -0.5) * (pred - gt)


def umap_loss_single(center_embedding, neighbor_embeddings, high_sims, a, b, sigma=None):
    cur_dist = np.linalg.norm(center_embedding - neighbor_embeddings, axis=-1)
    low_sims = convert_distance_to_probability(cur_dist, a, b)
    # print("high sims:", high_sims)
    # print("low sims:", low_sims)

    # 放大高维相似度
    # high_sims += 0.02
    # high_sims[high_sims > 1] = 1

    # CE Loss
    # loss = np.mean(-(high_sims * np.log(low_sims) + (1 - high_sims) * np.log(1 - low_sims)))

    # MSE Loss
    # sigma_sim = convert_distance_to_probability(sigma, a, b)
    # mse_indices = np.argwhere(low_sims >= sigma_sim).squeeze()
    # loss = np.sum(np.square(high_sims[mse_indices] - low_sims[mse_indices]))
    # loss = np.mean(np.square(low_sims - high_sims))

    # MAE Loss
    # TODO: 点与点之间挨的太近了。这是因为MSE Loss和MAE Loss都只考虑了点的相似性关系，但是没有考虑不相似关系。
    # 交叉熵考虑了这一点，但是为什么优化的不好呢？其实就是因为相似和不相似关系冲突了。
    # print(high_sims)
    loss = np.mean(np.abs(low_sims - high_sims))
    # print("loss", loss)

    # Huber Loss
    # assert sigma is not None
    # sigma_sim = convert_distance_to_probability(sigma, a, b)
    # mae_indices = np.where(low_sims < sigma_sim)[0]
    # if len(mae_indices) == 0:
    #     loss = 0.5 * np.mean(np.square(low_sims - high_sims))
    # else:
    #     mse_indices = np.where(low_sims >= sigma_sim)[0]
    #     loss = (0.5 * np.sum(np.square(low_sims[mse_indices] - high_sims[mse_indices])) + \
    #             np.sum(sigma_sim * np.abs(low_sims[mae_indices] - high_sims[mae_indices]) -
    #                    0.5 * np.square(sigma_sim))) / neighbor_embeddings.shape[0]

    return loss


# 直接模仿模型训练时的负例采样行为采样负例，然后基于相同的t值计算样本与每个邻居点的相似度
# @jit
def nce_loss_single(center_embedding, neighbor_embeddings, neg_embeddings, a, b, t=0.15):
    # 为什么不能将这部分计算抽离出去。是因为每次嵌入更新后都需要利用该公式再次计算，所以计算不是重复的。
    center_embedding = center_embedding[np.newaxis, :]
    neighbor_num = neighbor_embeddings.shape[0]

    # 1*(k+m)
    dist = cdist(center_embedding, np.concatenate([neighbor_embeddings, neg_embeddings], axis=0))
    # 1*(k+m)
    umap_sims = convert_distance_to_probability(dist, a, b)

    pos_umap_sims = umap_sims[:, :neighbor_num]
    neg_umap_sims = umap_sims[:, neighbor_num:]
    # k*(1+m)
    total_umap_sims = np.concatenate([pos_umap_sims.T, np.repeat(neg_umap_sims, neighbor_embeddings.shape[0], 0)],
                                     axis=1) / t
    # k*1, total_neg_umap_sims也可以直接使用模型训练时的平均值
    pos_nce_sims = (total_umap_sims[:, 0]) / np.sum(total_umap_sims, axis=1)

    loss = np.mean(-np.log(pos_nce_sims))
    return loss


def simple_nce_loss_single(center_embedding, neighbor_embeddings, neg_embeddings, a, b, t=0.15):
    center_embedding = center_embedding[np.newaxis, :]
    neighbor_num = neighbor_embeddings.shape[0]

    # 1*(k+m)
    dist = cdist(center_embedding, np.concatenate([neighbor_embeddings, neg_embeddings], axis=0))
    # 1*(k+m)
    umap_sims = convert_distance_to_probability(dist, a, b)

    pos_umap_sims = umap_sims[:, :neighbor_num]
    neg_umap_sims = umap_sims[:, neighbor_num:]
    # k*(1+m)
    loss = -np.mean(pos_umap_sims) + 1.2 * np.mean(neg_umap_sims)
    return loss


def simple_nce_loss_grad(center_embedding, neighbor_embeddings, neg_embeddings, a, b, t=0.15):
    center_embedding = center_embedding[np.newaxis, :]
    neighbor_num = neighbor_embeddings.shape[0]

    # 1*(k+m)
    dist = cdist(center_embedding, np.concatenate([neighbor_embeddings, neg_embeddings], axis=0))
    # 1*(k+m)
    umap_sims = convert_distance_to_probability(dist, a, b)

    pos_umap_sims = umap_sims[:, :neighbor_num]
    neg_umap_sims = umap_sims[:, neighbor_num:]
    # grad_per =\


# @jit
def tsne_grad(center_embedding, high_sims, neighbor_embeddings, k):
    center_embedding = center_embedding[np.newaxis, :]
    dists = cdist(center_embedding, neighbor_embeddings) ** 2
    a = (1 + dists) ** -1
    f = 1 / (1 + dists) ** (0.5 + k/10)
    q = f / np.sum(f)
    grad = (1 + k/5) * np.sum(a * (high_sims - q) * (center_embedding - neighbor_embeddings).T, axis=1)
    return grad
