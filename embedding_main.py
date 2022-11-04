import os
import random
import time
import numpy as np
import h5py
import scipy
import torch
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from dataset.warppers import StreamingDatasetWrapper
from model.dr_models.ModelSets import MODELS
from model.scdr.dependencies.experiment import position_vis
from model.scdr.model_trainer import IncrementalCDREx
from model_update_main import IndicesGenerator, query_knn
from utils.common_utils import time_stamp_to_date_time_adjoin, get_config
from utils.constant_pool import ConfigInfo
from utils.metrics_tool import Metric, knn_score
from sklearn.manifold import TSNE
from utils.umap_utils import find_ab_params, convert_distance_to_probability
from scipy import optimize
from numba import jit

device = "cuda:0"


# 从之前数据中选择一个与新数据邻域相似度模式最相似、且嵌入质量最好的一个点，模仿这个嵌入
def imitate_previous(center_knn_dists, knn_dists, low_neighbor_dists, with_e_quality=True):
    normed_knn_dists = (knn_dists - knn_dists[:, 0][:, np.newaxis]) / (
            knn_dists[:, -1][:, np.newaxis] - knn_dists[:, 0][:, np.newaxis])
    normed_c_knn_dists = (center_knn_dists - center_knn_dists[:, 0]) / (
            center_knn_dists[:, -1] - center_knn_dists[:, 0])
    center_neighbor_pattern = normed_c_knn_dists[:, 1:] - normed_c_knn_dists[:, :-1]
    neighbor_patterns = normed_knn_dists[:, 1:] - normed_knn_dists[:, :-1]
    neighbor_similarities = np.linalg.norm(center_neighbor_pattern - neighbor_patterns, axis=1)

    mean_neighbor_dists = np.mean(low_neighbor_dists)
    std_neighbor_dists = np.std(low_neighbor_dists)

    if not with_e_quality:
        final_idx = int(np.argmax(neighbor_similarities))
        return low_neighbor_dists[final_idx], mean_neighbor_dists + 2 * std_neighbor_dists

    min_low_neighbor_dists = np.min(low_neighbor_dists, axis=1)[:, np.newaxis]
    normed_low_dists = (low_neighbor_dists - min_low_neighbor_dists) / \
                       (np.max(low_neighbor_dists, axis=1)[:, np.newaxis] - min_low_neighbor_dists)
    embedding_qualities = np.sum(normed_low_dists, axis=1)

    min_e_quality = np.min(embedding_qualities)
    normed_e_qualities = (embedding_qualities - min_e_quality) / (np.max(embedding_qualities) - min_e_quality)

    final_idx = int(np.argmax(normed_e_qualities + neighbor_similarities))
    return low_neighbor_dists[final_idx], mean_neighbor_dists + 1 * std_neighbor_dists


def high_sims2low_dists_linear(x, a, b):
    y = a * x + b
    return y


def high_sims2low_dists_square(x, a, b, c):
    y = a * (x ** 2) + b * x + c
    return y


def trust_single(high_nn_indices, high_knn_indices, low_knn_indices, k):
    U = np.setdiff1d(low_knn_indices, high_knn_indices)
    N, n = high_nn_indices.shape
    sum_j = 0
    for j in range(U.shape[0]):
        high_rank = np.where(high_nn_indices.squeeze() == U[j])[0][0]
        sum_j += high_rank - k
    return 1 - (2 / (N * k * (2 * n - 3 * k - 1)) * sum_j)


def cont_single(low_nn_indices, high_knn_indices, low_knn_indices, k):
    V = np.setdiff1d(high_knn_indices, low_knn_indices)
    N, n = low_nn_indices.shape
    sum_j = 0
    for j in range(V.shape[0]):
        low_rank = np.where(low_nn_indices.squeeze() == V[j])[0][0]
        sum_j += low_rank - k
    return 1 - (2 / (N * k * (2 * n - 3 * k - 1)) * sum_j)


def neighbor_hit_single(labels, gt_label, low_knn_indices):
    pred_labels = labels[low_knn_indices]
    return np.sum(pred_labels == gt_label) / low_knn_indices.shape[1]


def dists_loss(center_embedding, neighbor_embeddings):
    cur_dist = np.linalg.norm(center_embedding - neighbor_embeddings, axis=-1)
    return np.mean(cur_dist)


def optimize_single_umap(center_embedding, neighbors_embeddings, target_sims, a, b, sigma=None):
    # sta = time.time()
    res = scipy.optimize.minimize(umap_loss_single, center_embedding, method="BFGS", jac=mae_loss_gard,
                                  args=(neighbors_embeddings, target_sims, a, b, sigma),
                                  options={'gtol': 1e-5, 'disp': False, 'return_all': False})
    # print("optimize cost:", time.time() - sta)
    return res.x


def optimize_single_nce(center_embedding, neighbors_embeddings, neg_embeddings, a, b, t=0.15):
    # sta = time.time()
    # eps参数控制的就是差分近似时候的步长，默认值非常小，是1e-8。这个值越小越快. 可能是因为越小就越容易满足最小误差。
    res = scipy.optimize.minimize(nce_loss_single, center_embedding, method="BFGS", jac=None,
                                  args=(neighbors_embeddings, neg_embeddings, a, b, t),
                                  options={'gtol': 1e-5, 'disp': False, 'return_all': False, 'eps': 1e-10})
    # print("optimize cost:", time.time() - sta)
    return res.x


count = 0

changed_indices = []
changed_indices_knn_dist_list = np.zeros(shape=2500, dtype=object)
avg_dist = []


def optimize_multiple(optimize_indices, knn_indices, total_embeddings, target_sims, anchor_position,
                      replaced_neighbor_indices, replaced_sims, a, b, local_move_mask, neg_nums=50,
                      nce_opt_update_thresh=10):
    # 使用BFGS算法进行优化
    neg_indices = random.sample(list(np.arange(total_embeddings.shape[0])), neg_nums)

    for i, item in enumerate(optimize_indices):
        if local_move_mask[i]:
            anchor_sims = target_sims[item, anchor_position[i]] / np.sum(target_sims[item])
            back = replaced_sims[i] * (total_embeddings[replaced_neighbor_indices[i]] - total_embeddings[item])
            total_embeddings[item] -= back

            move = anchor_sims * (total_embeddings[-1] - total_embeddings[item])
            total_embeddings[item] = total_embeddings[item] + move
        else:
            neg_embeddings = total_embeddings[neg_indices]
            optimized_e = optimize_single_nce(total_embeddings[item][np.newaxis, :], total_embeddings[knn_indices[item]],
                                              neg_embeddings, a, b)
            update_step = optimized_e - total_embeddings[item]
            # TODO:不像参数化方法，这种非参方法对NCE损失的鲁棒性比较差
            update_step[update_step > nce_opt_update_thresh] = 0
            update_step[update_step < -nce_opt_update_thresh] = -0
            total_embeddings[item] += update_step

    return total_embeddings[optimize_indices]

    # 在局部结构发生变化的方向上进行修正，跟原嵌入比较接近
    # 被替换的都是低相似度的，新数据都是高相似度的，就导致嵌入严重偏向新数据的嵌入方向。
    # num = len(optimize_indices)
    # anchor_sims = target_sims[optimize_indices, anchor_position] / np.sum(target_sims[optimize_indices], axis=1)
    # move = anchor_sims[:, np.newaxis] * np.repeat(total_embeddings[-1][np.newaxis, :], num, 0)
    # back = replaced_sims[:, np.newaxis] * total_embeddings[replaced_neighbor_indices]
    # step = move - back
    # thresh = 0.1
    # step[step > thresh] = thresh
    # step[step < -thresh] = -thresh
    # updated_embedding = total_embeddings[optimize_indices] + 0.001 * step

    # 使用所有邻居的插值
    # n_neighbors = knn_indices.shape[1]
    # changed_knn_indices = knn_indices[optimize_indices]
    # neighbor_embeddings = np.reshape(total_embeddings[np.ravel(changed_knn_indices)],
    #                                  (len(optimize_indices), n_neighbors, -1))
    # optimize_sims = target_sims[optimize_indices] / np.sum(target_sims[optimize_indices], axis=1)[:, np.newaxis]
    # updated_embedding = np.sum(neighbor_embeddings * np.repeat(optimize_sims[:, :, np.newaxis],
    #                                                            total_embeddings.shape[1], -1), axis=1)

    # return updated_embedding


def optimize_all_waits(indices, knn_indices, total_embeddings, a, b, neg_num=50, nce_opt_update_thresh=10):
    neg_indices = random.sample(list(np.arange(total_embeddings.shape[0])), neg_num)
    for i, item in enumerate(indices):
        neg_embeddings = total_embeddings[neg_indices]
        optimized_e = optimize_single_nce(total_embeddings[item][np.newaxis, :],
                                          total_embeddings[knn_indices[item]],
                                          neg_embeddings, a, b)
        update_step = optimized_e - total_embeddings[item]
        # TODO:不像参数化方法，这种非参方法对NCE损失的鲁棒性比较差
        update_step[update_step > nce_opt_update_thresh] = 0
        update_step[update_step < -nce_opt_update_thresh] = -0
        total_embeddings[item] += update_step

    return total_embeddings[indices]


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
    # grad_per =


def incremental_cdr_pipeline():
    def _print_some(pre_text, nc_list, ar_list):
        mean_nc = np.mean(nc_list)
        std_nc = np.std(nc_list)
        mean_ar = np.mean(ar_list)
        std_ar = np.std(ar_list)
        print(pre_text, " Mean NC: %.2f Std NC: %.2f Mean AR: %.2f Std AR: %.2f" % (mean_nc, std_nc, mean_ar, std_ar))

    cfg.merge_from_file(CONFIG_PATH)
    clr_model = MODELS[METHOD_NAME](cfg, device=device)
    t_log_path = os.path.join(RESULT_SAVE_DIR, "logs.txt")
    experimenter = IncrementalCDREx(clr_model, cfg.exp_params.dataset, cfg, RESULT_SAVE_DIR, CONFIG_PATH, shuffle=True,
                                    device=device, log_path=t_log_path)
    n_neighbors = experimenter.n_neighbors
    a, b = find_ab_params(1.0, MIN_DIST)

    with h5py.File(os.path.join(ConfigInfo.DATASET_CACHE_DIR, "{}.h5".format(experimenter.dataset_name)), "r") as hf:
        total_data = np.array(hf['x'])
        total_labels = np.array(hf['y'], dtype=int)

    # 0.需要创建一个索引生成器，根据不同的条件生成一批批的索引列表
    indices_generator = IndicesGenerator(total_labels)

    # n_step, cls_num = 4, 5
    # batch_indices = indices_generator.no_dis_change(cls_num, n_step)

    total_manifold_num, initial_manifold_num = 6, 3
    n_step = total_manifold_num - initial_manifold_num + 1
    batch_indices = indices_generator.mixing(initial_manifold_num, total_manifold_num)
    # initial_cls_num, after_cls_num_list = 3, [2, 2, 3]
    # initial_cls_num, after_cls_num_list = 3, [1, 2, 3]
    # n_step = len(after_cls_num_list) + 1
    # batch_indices = indices_generator.heavy_dis_change(initial_cls_num, after_cls_num_list)

    # 1.需要创建一个数据整合器，用于从旧数据中采样代表点数据
    initial_data = total_data[batch_indices[0]]
    initial_labels = total_labels[batch_indices[0]]
    fitted_manifolds = set(np.unique(initial_labels))
    stream_dataset = StreamingDatasetWrapper(initial_data, initial_labels, experimenter.batch_size,
                                             experimenter.n_neighbors)
    ckpt_path = r"results\embedding_ex\comparison_ex\only_model\initial\CDR_100.pth.tar"
    # ckpt_path = None
    initial_embeddings = experimenter.first_train(stream_dataset, INITIAL_EPOCHS, ckpt_path)
    position_vis(stream_dataset.total_label, os.path.join(RESULT_SAVE_DIR, "0.jpg"), initial_embeddings,
                 title="Initial Embeddings")

    # predict_low_dists(initial_embeddings, stream_dataset)

    experimenter.active_incremental_learning()
    fitted_indices = batch_indices[0]
    total_neighbor_cover_list = [[], []]
    total_avg_rank_list = [[], []]
    before_t_list = []
    after_t_list = []
    before_c_list = []
    after_c_list = []
    before_nh_list = []
    after_nh_list = []

    new_data_avg_knn_dist = []
    changed_data_avg_knn_dist = []

    self_optimize_time = 0
    s_neighbor_time = 0
    o_neighbor_time = 0
    knn_update_time = 0
    infer_time = 0
    knn_cal_time = 0
    eval_time = 0
    vis_time = 0
    data_add_time = 0
    embedding_concat_time = 0
    # print(batch_indices)

    pre_neighbor_mean_dist = np.mean(stream_dataset.get_knn_dists())
    pre_neighbor_std_dist = np.std(stream_dataset.get_knn_dists())
    print(pre_neighbor_mean_dist, pre_neighbor_std_dist)
    pre_low_neighbor_embedding_dists = np.linalg.norm(experimenter.pre_embeddings[:, np.newaxis, :] - np.reshape(
        experimenter.pre_embeddings[np.ravel(stream_dataset.get_knn_indices())],
        (experimenter.pre_embeddings.shape[0], K, -1)), axis=-1)
    pre_neighbor_embedding_m_dist = np.mean(pre_low_neighbor_embedding_dists)
    pre_neighbor_embedding_s_dist = np.std(pre_low_neighbor_embedding_dists)
    print(pre_neighbor_embedding_m_dist)
    print(pre_neighbor_embedding_s_dist)
    new_changed_num = 0
    old_changed_num = 0

    wait_for_optimize_meta = None
    total_pre_optimize = 0
    total_skip_opt = 0

    total_sta = time.time()
    for i in range(1, n_step):
        cur_batch_data = total_data[batch_indices[i]]
        cur_batch_num = len(batch_indices[i])
        batch_neighbor_cover_list = [[], []]
        batch_avg_rank_list = [[], []]

        for j in range(cur_batch_num):
            cur_data_idx = batch_indices[i][j]
            cur_data_seq_idx = experimenter.pre_embeddings.shape[0]
            cur_data = cur_batch_data[j][np.newaxis, :]
            cur_label = total_labels[cur_data_idx]
            sta = time.time()
            cur_embedding = experimenter.cal_lower_embeddings(cur_data)[np.newaxis, :]
            infer_time += time.time() - sta
            previous_knn_indices, previous_knn_dists = stream_dataset.get_knn_indices(), stream_dataset.get_knn_dists()

            sta = time.time()
            knn_indices, knn_dists, sort_indices = \
                query_knn(cur_data, np.concatenate([stream_dataset.total_data, cur_data], axis=0), k=K,
                          return_indices=True)
            new_data_avg_knn_dist.append(np.mean(knn_dists))
            knn_cal_time += time.time() - sta

            # before_mean_dist = np.mean(low_knn_dists)

            # 比较简单高效的方法就是直接基于neighbor cover来判断，低于之前的均值-sigma就使用插值法来确定初始嵌入
            dis_change = int(cur_label) not in fitted_manifolds
            # dis_change = neighbor_cover < 0.21 - 0.14   # mean - alpha * std

            # dists2pre_embeddings = cdist(cur_embedding, experimenter.pre_embeddings)
            # sort_indices = np.argsort(dists2pre_embeddings).squeeze()
            # average_rank = 0
            # for item in knn_indices[0]:
            #     tmp_idx = int(np.argwhere(sort_indices == item).squeeze())
            #     average_rank += tmp_idx

            # DEBUG
            # dis_change = False

            sta = time.time()
            stream_dataset.add_new_data(cur_data, knn_indices, knn_dists, cur_label)
            data_add_time += time.time() - sta
            # TODO：这里更新邻居权重非常耗时！

            sta = time.time()
            stream_dataset.update_knn_graph(len(fitted_indices), cur_data, [0], update_similarity=False,
                                            symmetric=False)
            knn_update_time += time.time() - sta

            sta = time.time()
            low_knn_indices, low_knn_dists, low_nn_indices = \
                query_knn(cur_embedding, np.concatenate([experimenter.pre_embeddings, cur_embedding], axis=0), k=K,
                          return_indices=True)
            knn_cal_time += time.time() - sta

            final_embedding = cur_embedding

            if dis_change:
                cur_neighbor_embeddings = experimenter.pre_embeddings[knn_indices[0]]
                # ====================================1. 只对新数据本身的嵌入进行更新=======================================
                if OPTIMIZE_NEW_DATA_EMBEDDING:
                    sta = time.time()
                    # 这里选择出来的点，邻域相似度都太高了，就导致新数据的嵌入非常紧凑。这是因为考虑嵌入质量时就是使用紧密程度作为参考的。
                    neighbor_sims = stream_dataset.raw_knn_weights[-1]
                    normed_neighbor_sims = neighbor_sims / np.sum(neighbor_sims)
                    cur_initial_embedding = np.sum(normed_neighbor_sims[:, np.newaxis] *
                                                   cur_neighbor_embeddings, axis=0)[np.newaxis, :]
                    # final_embedding = cur_initial_embedding.squeeze()

                    # sigma = 0
                    # low_neighbor_dists = np.linalg.norm(experimenter.pre_embeddings[:, np.newaxis, :] -
                    #                                     np.reshape(
                    #                                         experimenter.pre_embeddings[np.ravel(previous_knn_indices)],
                    #                                         (experimenter.pre_embeddings.shape[0], K, -1)), axis=-1)
                    # target_dists, sigma = imitate_previous(knn_dists, previous_knn_dists, low_neighbor_dists,
                    #                                        with_e_quality=True)
                    # imitate_sims = convert_distance_to_probability(target_dists, a, b)
                    # final_embedding = optimize_single_umap(cur_initial_embedding, cur_neighbor_embeddings,
                    #                                        imitate_sims, a, b, sigma)

                    neg_nums = 20
                    sta = time.time()
                    if np.mean(knn_dists) <= pre_neighbor_mean_dist + 10 * pre_neighbor_std_dist:
                        neg_embeddings = experimenter.pre_embeddings[
                            random.sample(list(np.arange(experimenter.pre_embeddings.shape[0])), neg_nums)]
                        final_embedding = optimize_single_nce(cur_initial_embedding, cur_neighbor_embeddings,
                                                              neg_embeddings, a, b)
                        final_embedding = final_embedding[np.newaxis, :]
                    else:
                        final_embedding = cur_initial_embedding

                    self_optimize_time += time.time() - sta
                # =====================================================================================================

                # ====================================2. 对新数据及其邻居点的嵌入进行更新====================================
                # TODO: 无论新数据是否来自新的流形，更新kNN发生变化的旧数据嵌入都是必要的
                # TODO：对共享邻居和单向邻居的嵌入优化都非常耗时
                sta = time.time()
                # 注意！并不是所有邻居点都跟他互相是邻居！
                if OPTIMIZE_SHARED_NEIGHBORS:
                    pre_left_wait_meta = None
                    time_wait_optimized_indices = []
                    if SKIP_OPT:
                        if wait_for_optimize_meta is not None:
                            time_wait_optimized_indices = \
                                np.where(time.time() - wait_for_optimize_meta[:, 1] >= INTER_THRESH)[0]
                            if len(time_wait_optimized_indices) > 0:
                                total_pre_optimize += len(time_wait_optimized_indices)
                                # print("pre optimize", total_pre_optimize)
                                experimenter.pre_embeddings[time_wait_optimized_indices] = \
                                    optimize_all_waits(time_wait_optimized_indices, stream_dataset.get_knn_indices(),
                                                       np.concatenate([experimenter.pre_embeddings, final_embedding],
                                                                      axis=0), a, b)

                    shared_neighbor_meta, oneway_neighbor_meta = stream_dataset.get_pre_neighbor_changed_positions(-1)
                    shared_neighbor_indices = shared_neighbor_meta[:, 0] if len(shared_neighbor_meta) > 0 else []
                    replaced_raw_weights_s = stream_dataset.get_replaced_raw_weights(shared=True)
                    replaced_indices_s = shared_neighbor_meta[:, 2] if len(shared_neighbor_meta) > 0 else []
                    anchor_position_shared = shared_neighbor_meta[:, 1] if len(shared_neighbor_meta) > 0 else []

                    oneway_neighbor_indices = np.setdiff1d(oneway_neighbor_meta[:, 0], [cur_data_seq_idx]) \
                        if len(oneway_neighbor_meta) > 0 else []
                    replaced_raw_weights_o = stream_dataset.get_replaced_raw_weights(shared=False)
                    replaced_indices_o = oneway_neighbor_meta[:, 2] if len(oneway_neighbor_indices) > 0 else []
                    anchor_position_oneway = oneway_neighbor_meta[:, 1] if len(oneway_neighbor_indices) > 0 else []

                    neighbor_changed_indices = np.concatenate([shared_neighbor_indices, oneway_neighbor_indices]).astype(int)
                    replaced_raw_weights = np.concatenate([replaced_raw_weights_s, replaced_raw_weights_o], axis=0)
                    replaced_indices = np.concatenate([replaced_indices_s, replaced_indices_o], axis=0)
                    anchor_positions = np.concatenate([anchor_position_shared, anchor_position_oneway], axis=0)

                    if len(neighbor_changed_indices) > 0:
                        cur_local_move_thresh = pre_neighbor_embedding_m_dist
                        cur_local_move_mask = np.linalg.norm(experimenter.pre_embeddings[neighbor_changed_indices] -
                                                             final_embedding, axis=-1) < cur_local_move_thresh

                        # print("origin:", len(neighbor_changed_indices))
                        # print("origin opt num:", np.sum(~cur_local_move_mask))

                        cur_optimize_thresh = pre_neighbor_mean_dist + 1 * pre_neighbor_std_dist
                        cur_optimize_mask = np.mean(stream_dataset.get_knn_dists()[neighbor_changed_indices],
                                                    axis=1) < cur_optimize_thresh
                        embedding_update_mask = (cur_local_move_mask + cur_optimize_mask) > 0

                        cur_update_indices = neighbor_changed_indices[embedding_update_mask]
                        cur_wait_indices = neighbor_changed_indices[~embedding_update_mask]

                        total_skip_opt += len(cur_wait_indices)
                        # print("wait optimize:", total_skip_opt)

                        neighbor_changed_indices = neighbor_changed_indices[embedding_update_mask]
                        anchor_positions = anchor_positions[embedding_update_mask].astype(int)
                        replaced_indices = replaced_indices[embedding_update_mask].astype(int)
                        replaced_raw_weights = replaced_raw_weights[embedding_update_mask]
                        cur_local_move_mask = cur_local_move_mask[embedding_update_mask]

                        # print("after:", len(neighbor_changed_indices))
                        # print("after opt num:", np.sum(~cur_local_move_mask))

                        t_sta = time.time()
                        updated_embeddings = \
                            optimize_multiple(neighbor_changed_indices, stream_dataset.get_knn_indices(),
                                              np.concatenate([experimenter.pre_embeddings, final_embedding], axis=0),
                                              stream_dataset.raw_knn_weights, anchor_positions,
                                              replaced_indices, replaced_raw_weights, a, b, cur_local_move_mask)
                        embedding_concat_time += time.time() - t_sta

                        experimenter.pre_embeddings[neighbor_changed_indices] = updated_embeddings

                        if SKIP_OPT:
                            if wait_for_optimize_meta is not None:
                                pre_left_wait_meta_indices = np.setdiff1d(np.arange(wait_for_optimize_meta.shape[0]),
                                                                          time_wait_optimized_indices)
                                pre_left_wait_meta = wait_for_optimize_meta[pre_left_wait_meta_indices]
                                neighbor_wait_optimized_indices, position_1, _ = \
                                    np.intersect1d(pre_left_wait_meta[:, 0], cur_update_indices, return_indices=True)

                                pre_left_meta_indices = np.setdiff1d(np.arange(len(pre_left_wait_meta_indices)),
                                                                     position_1)
                                pre_left_wait_meta = pre_left_wait_meta[pre_left_meta_indices]

                            if pre_left_wait_meta is not None:
                                if len(pre_left_wait_meta.shape) < 2:
                                    pre_left_wait_meta = pre_left_wait_meta[np.newaxis, :]

                                cur_wait_indices = np.setdiff1d(cur_wait_indices, pre_left_wait_meta[:, 0])
                                cur_wait_meta = np.zeros(shape=(len(cur_wait_indices), 2))
                                cur_wait_meta[:, 0] = cur_wait_indices
                                cur_wait_meta[:, 1] = time.time()

                                wait_for_optimize_meta = np.concatenate([pre_left_wait_meta, cur_wait_meta], axis=0)
                            else:
                                wait_for_optimize_meta = np.zeros(shape=(len(cur_wait_indices), 2))
                                wait_for_optimize_meta[:, 0] = cur_wait_indices
                                wait_for_optimize_meta[:, 1] = time.time()

                s_neighbor_time += time.time() - sta
                # ======================================================================================================

            # 之前的一些数据的嵌入位置不再是模型嵌入的了，所以导致计算的kNN也发生了变化，使得kNN不一样。
            sta = time.time()
            after_low_knn_indices, after_low_knn_dists, after_low_nn_indices = \
                query_knn(final_embedding, np.concatenate([experimenter.pre_embeddings, final_embedding], axis=0), k=K,
                          return_indices=True)
            before_trust = trust_single(sort_indices, knn_indices, low_knn_indices, n_neighbors)
            after_trust = trust_single(sort_indices, knn_indices, after_low_knn_indices, n_neighbors)
            before_cont = cont_single(low_nn_indices, knn_indices, low_knn_indices, n_neighbors)
            after_cont = cont_single(after_low_nn_indices, knn_indices, after_low_knn_indices, n_neighbors)
            before_nh = neighbor_hit_single(stream_dataset.total_label, cur_label, low_knn_indices)
            after_nh = neighbor_hit_single(stream_dataset.total_label, cur_label, after_low_knn_indices)
            # print(before_trust, after_trust)

            rec_idx = 1 if dis_change else 0
            before_t_list.append(before_trust)
            after_t_list.append(after_trust)
            before_c_list.append(before_cont)
            after_c_list.append(after_cont)
            before_nh_list.append(before_nh)
            after_nh_list.append(after_nh)

            eval_time += time.time() - sta

            sta = time.time()
            fitted_indices.append(cur_data_idx)
            experimenter.pre_embeddings = np.concatenate([experimenter.pre_embeddings, final_embedding], axis=0)
            embedding_concat_time += time.time() - sta

            # sta = time.time()
            # for t in stream_dataset.cur_neighbor_changed_indices:
            #     high_sort_indices = np.argsort(cdist(stream_dataset.total_data[t][np.newaxis, :], stream_dataset.total_data), axis=1)
            #     before_knn_indices = query_knn(tmp_total_embeddings[t][np.newaxis, :], tmp_total_embeddings, k=K)[0]
            #     after_knn_indices = query_knn(tmp_total_embeddings[t][np.newaxis, :], experimenter.pre_embeddings, k=K)[0]
            #     before_trust = trust(high_sort_indices, stream_dataset.get_knn_indices()[t], before_knn_indices, n_neighbors)
            #     after_trust = trust(high_sort_indices, stream_dataset.get_knn_indices()[t], after_knn_indices, n_neighbors)
            #     # before_cont =
            #     # TODO: 添加continuity的度量
            #     before_t_list.append(before_trust)
            #     after_t_list.append(after_trust)
            # eval_time += time.time() - sta

        # _print_some("No Change!", batch_neighbor_cover_list[0], batch_avg_rank_list[0])
        # _print_some("Change!", batch_neighbor_cover_list[1], batch_avg_rank_list[1])

        # total_neighbor_cover_list[0].extend(batch_neighbor_cover_list[0])
        # total_avg_rank_list[0].extend(batch_avg_rank_list[0])
        # total_neighbor_cover_list[1].extend(batch_neighbor_cover_list[1])
        # total_avg_rank_list[1].extend(batch_avg_rank_list[1])
        sta = time.time()
        position_vis(stream_dataset.total_label, os.path.join(RESULT_SAVE_DIR, "{}.jpg".format(i)),
                     experimenter.pre_embeddings, title="Step {} Embeddings".format(i))
        vis_time += time.time() - sta

    if wait_for_optimize_meta is not None:
        if len(wait_for_optimize_meta.shape) < 2:
            left_indices = [int(wait_for_optimize_meta[0])]
        else:
            left_indices = wait_for_optimize_meta[:, 0].astype(int)
        print("left", len(left_indices))
        experimenter.pre_embeddings[left_indices] = optimize_all_waits(left_indices, stream_dataset.get_knn_indices(),
                                                                       experimenter.pre_embeddings, a, b)

    sta = time.time()
    position_vis(stream_dataset.total_label, os.path.join(RESULT_SAVE_DIR, "final.jpg"),
                 experimenter.pre_embeddings, title="Final Embeddings")
    vis_time += time.time() - sta

    # plt.title("Pre Mean: %.4f" % pre_neighbor_mean_dist)
    # plt.violinplot(changed_data_avg_knn_dist, showmeans=True)
    # plt.show()

    sta = time.time()
    metric_tool = Metric(experimenter.dataset_name, stream_dataset.total_data, stream_dataset.total_label, None, None,
                         k=K)
    total_trust = metric_tool.metric_trustworthiness(K, experimenter.pre_embeddings)
    total_cont = metric_tool.metric_continuity(K, experimenter.pre_embeddings)
    total_nh = metric_tool.metric_neighborhood_hit(K, experimenter.pre_embeddings)
    total_knn = knn_score(experimenter.pre_embeddings, stream_dataset.total_label, K,
                          knn_indices=metric_tool.low_knn_indices)
    eval_time += time.time() - sta

    print("old change: %d new change: %d" % (old_changed_num, new_changed_num))

    # _print_some("No Change!", total_neighbor_cover_list[0], total_avg_rank_list[0])
    # _print_some("Change!", total_neighbor_cover_list[1], total_avg_rank_list[1])
    # TODO：需要对比一下直接使用模型嵌入，和基于牛顿法优化的嵌入质量差异
    print("Before Trust: %.4f Before Cont: %.4f Before NH: %.4f" % (
        np.mean(before_t_list), np.mean(before_c_list), np.mean(before_nh_list)))
    print("After Trust: %.4f After Cont: %.4f After NH: %.4f" % (
        np.mean(after_t_list), np.mean(after_c_list), np.mean(after_nh_list)))
    print("Total Trust: %.4f Total Cont: %.4f Total NH: %.4f Total kNN: %.4f" % (
        total_trust, total_cont, total_nh, total_knn))
    print("Total: %.4f Eval: %.4f kNN Update: %.4f Infer: %.4f kNN Cal: %.4f Vis: %.4f Self: %.4f Shared: %.4f "
          "Oneway: %.4f Data Add: %.4f Embedding Concat: %.4f" % (time.time() - total_sta, eval_time, knn_update_time,
                                                                  infer_time, knn_cal_time, vis_time,
                                                                  self_optimize_time, s_neighbor_time, o_neighbor_time,
                                                                  data_add_time, embedding_concat_time))
    print("Data Num:", len(new_data_avg_knn_dist), " Key Time:", knn_update_time + infer_time +
          knn_cal_time + self_optimize_time + s_neighbor_time + o_neighbor_time)


if __name__ == '__main__':
    with torch.cuda.device(0):
        time_step = time_stamp_to_date_time_adjoin(time.time())
        RESULT_SAVE_DIR = r"results/embedding_ex/{}".format(time_step)
        os.mkdir(RESULT_SAVE_DIR)
        METHOD_NAME = "LwF_CDR"
        CONFIG_PATH = "configs/ParallelSCDR.yaml"

        OPTIMIZE_NEW_DATA_EMBEDDING = True
        OPTIMIZE_SHARED_NEIGHBORS = True
        OPTIMIZE_ONEWAY_NEIGHBORS = True
        MIN_DIST = 0.1
        INITIAL_EPOCHS = 100
        K = 10
        INTER_THRESH = 1.0
        SKIP_OPT = False

        cfg = get_config()
        cfg.merge_from_file(CONFIG_PATH)

        incremental_cdr_pipeline()
