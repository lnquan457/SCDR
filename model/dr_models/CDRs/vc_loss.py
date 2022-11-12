import random
import time

from utils.math_utils import compute_rank_correlation
import torch
from audtorch.metrics.functional import pearsonr


def temporal_steady_loss(preserve_rank=False, preserve_positions=False, preserve_shape=False,
                         rank_weights=1.0, position_weight=1.0, shape_weight=1.0, rep_embeddings=None,
                         rep_neighbors_embeddings=None,
                         pre_rep_embeddings=None, cluster_indices=None, exclude_indices=None,
                         pairwise_dist=None, pre_pairwise_dist=None, pre_rep_neighbors_embeddings=None,
                         no_change_indices=None, steady_weights=None, neighbor_steady_weights=None):
    t_sta = time.time()
    rank_relation_loss = torch.tensor(0)
    position_relation_loss = torch.tensor(0)
    shape_loss = torch.tensor(0)
    cluster_num = cluster_indices.shape[0]

    if preserve_rank and cluster_num > 1:
        sta = time.time()
        rank_relation_loss = cal_rank_relation_loss(rep_embeddings, pre_rep_embeddings, cluster_indices,
                                                    exclude_indices, pairwise_dist, pre_pairwise_dist)
        print("     rank", time.time() - sta)

    if preserve_positions and cluster_num > 1:
        sta = time.time()
        c_indices = [item[random.randint(0, len(item) - 1)] for item in cluster_indices]
        position_relation_loss = cal_position_relation_loss(rep_embeddings[c_indices], pre_rep_embeddings[c_indices])
        print("     positions", time.time() - sta)

    if preserve_shape and len(no_change_indices) > 0:
        sta = time.time()
        shape_loss = cal_shape_loss(rep_embeddings[no_change_indices], rep_neighbors_embeddings,
                                    pre_rep_embeddings[no_change_indices], pre_rep_neighbors_embeddings,
                                    steady_weights, neighbor_steady_weights)
        print("     shape:", time.time() - sta)

    # print("=============")
    # print("rank_relation_loss", rank_relation_loss.item())
    # print("position_relation_loss", position_relation_loss.item())
    # print("shape_loss", shape_loss.item())
    # print("=============")

    w_rank_relation_loss = rank_weights * rank_relation_loss
    w_position_relation_loss = position_weight * position_relation_loss
    w_shape_loss = shape_weight * shape_loss

    loss = w_rank_relation_loss + w_position_relation_loss + w_shape_loss
    print("     total ts:", time.time() - t_sta)
    return loss, w_rank_relation_loss, w_position_relation_loss, w_shape_loss


def cal_rank_relation_loss(rep_embeddings, pre_rep_embeddings):
    cluster_nums = rep_embeddings.shape[0]

    dists = torch.linalg.norm(rep_embeddings.unsqueeze(0).repeat(cluster_nums, 1, 1) -
                              rep_embeddings.unsqueeze(1).repeat(1, cluster_nums, 1), dim=-1)
    pre_dists = torch.linalg.norm(pre_rep_embeddings.unsqueeze(0).repeat(cluster_nums, 1, 1) -
                                  pre_rep_embeddings.unsqueeze(1).repeat(1, cluster_nums, 1), dim=-1)

    return -compute_rank_correlation(dists.cpu(), pre_dists.cpu())


def cal_position_relation_loss(cluster_center_embeddings: torch.Tensor,
                               pre_cluster_center_embeddings: torch.Tensor):
    cluster_nums = cluster_center_embeddings.shape[0]
    sims = torch.cosine_similarity(cluster_center_embeddings.unsqueeze(0).repeat(cluster_nums, 1, 1) -
                                   cluster_center_embeddings.unsqueeze(1).repeat(1, cluster_nums, 1),
                                   pre_cluster_center_embeddings.unsqueeze(0).repeat(cluster_nums, 1, 1) -
                                   pre_cluster_center_embeddings.unsqueeze(1).repeat(1, cluster_nums, 1), dim=-1)

    loss = -torch.mean(torch.square(sims))
    return loss


def cal_shape_loss(rep_embeddings, rep_neighbors_embeddings, pre_rep_embeddings, pre_rep_neighbors_embeddings,
                   steady_weights=None, neighbor_steady_weights=None):
    sims = torch.cosine_similarity(rep_embeddings - rep_neighbors_embeddings,
                                   pre_rep_embeddings - pre_rep_neighbors_embeddings, dim=-1)

    if steady_weights is not None and neighbor_steady_weights is not None:
        sims *= (steady_weights + neighbor_steady_weights) / 2

    # return torch.linalg.norm(sims_change)
    return -torch.mean(torch.square(sims))


def cal_space_expand_loss(rep_embeddings, pre_rep_embeddings, cluster_indices, exclude_indices,
                          pairwise_dists=None, pre_pairwise_dists=None, expand_rate=0.3):
    cluster_num = len(cluster_indices)
    rate_diff = torch.zeros(cluster_num)

    pairwise_dists = torch.cdist(rep_embeddings, rep_embeddings).cpu() if pairwise_dists is not None else pairwise_dists
    pre_pairwise_dists = torch.cdist(pre_rep_embeddings,
                                     pre_rep_embeddings).cpu() if pre_pairwise_dists is None else pre_pairwise_dists

    for i, item in enumerate(cluster_indices):
        other_cluster_indices = exclude_indices[i]
        change_rate = (pairwise_dists[item][:, other_cluster_indices] -
                       pre_pairwise_dists[item][:, other_cluster_indices]) / pre_pairwise_dists[item][:,
                                                                             other_cluster_indices]
        # print(pairwise_dists[0, other_cluster_indices] - pre_pairwise_dists[0, other_cluster_indices])
        rate_diff[i] = torch.mean(torch.square(expand_rate - change_rate))

    return torch.mean(rate_diff)
