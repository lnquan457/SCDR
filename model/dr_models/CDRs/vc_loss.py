from utils.math_utils import compute_rank_correlation
import torch
from audtorch.metrics.functional import pearsonr


def _cal_pairwise_dist_change(cur_embeddings, pre_embeddings, corr=False):
    rep_nums = cur_embeddings.shape[0]
    cur_rep_embedding_matrix = cur_embeddings.unsqueeze(1).repeat(1, rep_nums, 1)
    cur_dists = torch.norm(cur_rep_embedding_matrix - torch.transpose(cur_rep_embedding_matrix, 0, 1), dim=-1)

    pre_rep_embedding_matrix = pre_embeddings.unsqueeze(1).repeat(1, rep_nums, 1)
    pre_dists = torch.norm(pre_rep_embedding_matrix - torch.transpose(pre_rep_embedding_matrix, 0, 1), dim=-1)
    if corr:
        return compute_rank_correlation(cur_dists, pre_dists)
    return torch.mean(torch.norm(cur_dists - pre_dists))


def _visual_consistency_loss(cur_rep_embeddings, pre_rep_embeddings, position=False, cluster_indices=None,
                             exclude_indices=None, pre_pairwise_dist=None):
    if position:
        return torch.mean(torch.norm(cur_rep_embeddings - pre_rep_embeddings, dim=1))
    else:
        # return _cal_pairwise_dist_change(cur_rep_embeddings.cpu(), pre_rep_embeddings.cpu(), corr=False)
        # return _with_pearson_and_spearman_corr(cur_rep_embeddings.cpu(), pre_rep_embeddings.cpu(), cluster_indices,
        #                                        exclude_indices)
        return _with_pairwise_dist_change(cur_rep_embeddings.cpu(), pre_rep_embeddings.cpu(), cluster_indices,
                                          exclude_indices, pre_pairwise_dist.cpu())


def _with_pearson_and_spearman_corr(cur_embeddings, pre_embeddings, cluster_indices, exclude_indices):
    cluster_num = len(cluster_indices)
    inner_pearson_corr = torch.zeros(cluster_num)
    intra_spearman_corr = torch.zeros(cluster_num)

    cur_pairwise_dists = torch.cdist(cur_embeddings, cur_embeddings)
    pre_pairwise_dists = torch.cdist(pre_embeddings, pre_embeddings)

    for i, item in enumerate(cluster_indices):
        other_cluster_indices = exclude_indices[i]
        inner_pearson_corr[i] = torch.mean(pearsonr(cur_pairwise_dists[item][:, item],
                                                    pre_pairwise_dists[item][:, item])[:, 0])

        intra_spearman_corr[i] = compute_rank_correlation(cur_pairwise_dists[item][:, other_cluster_indices],
                                                          pre_pairwise_dists[item][:, other_cluster_indices])
    # print("pearson corr:", torch.mean(inner_pearson_corr).item())
    # print("spearman corr:", torch.mean(intra_spearman_corr).item())
    # return torch.mean(inner_pearson_corr) + torch.mean(intra_spearman_corr)
    return torch.mean(inner_pearson_corr)


def _with_pairwise_dist_change(cur_embeddings, pre_embeddings, cluster_indices, exclude_indices, pre_pairwise_dists=None):
    cluster_num = len(cluster_indices)
    inner_dist_change = torch.zeros(cluster_num)
    intra_dist_change = torch.zeros(cluster_num)
    intra_spearman_corr = torch.zeros(cluster_num)

    cur_pairwise_dists = torch.cdist(cur_embeddings, cur_embeddings)
    pre_pairwise_dists = torch.cdist(pre_embeddings, pre_embeddings) if pre_pairwise_dists is None else pre_pairwise_dists

    for i, item in enumerate(cluster_indices):
        other_cluster_indices = exclude_indices[i]
        inner_dist_change[i] = torch.mean(torch.norm(cur_pairwise_dists[item][:, item] -
                                                     pre_pairwise_dists[item][:, item], dim=-1))

        intra_dist_change[i] = torch.mean(torch.norm(cur_pairwise_dists[item][:, other_cluster_indices] -
                                                     pre_pairwise_dists[item][:, other_cluster_indices], dim=-1))

        intra_spearman_corr[i] = compute_rank_correlation(cur_pairwise_dists[item][:, other_cluster_indices],
                                                          pre_pairwise_dists[item][:, other_cluster_indices])

    # print("pearson corr:", torch.mean(inner_pearson_corr).item())
    # print("intra dist change:", torch.mean(intra_dist_change).item())
    # print("spearman corr:", torch.mean(intra_spearman_corr).item())
    # return torch.mean(inner_pearson_corr) + torch.mean(intra_spearman_corr)
    return torch.mean(inner_dist_change) - torch.mean(intra_spearman_corr)
    # return torch.mean(inner_dist_change)
