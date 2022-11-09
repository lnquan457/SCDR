import random

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


def temporal_steady_loss(preserve_rank=False, preserve_positions=False, preserve_shape=False, rank_weights=1,
                         position_weight=1, shape_weight=1, rep_embeddings=None,
                         pre_rep_embeddings=None, cluster_indices=None, exclude_indices=None, pre_pairwise_dist=None,
                         cluster_center_embeddings=None, pre_cluster_center_embeddings=None):
    # return _cal_pairwise_dist_change(cur_rep_embeddings.cpu(), pre_rep_embeddings.cpu(), corr=False)
    # return _with_pearson_and_spearman_corr(cur_rep_embeddings.cpu(), pre_rep_embeddings.cpu(), cluster_indices,
    #                                        exclude_indices)
    # return _with_pairwise_dist_change(cur_rep_embeddings.cpu(), pre_rep_embeddings.cpu(),
    #                                   cluster_indices, exclude_indices, pre_pairwise_dist.cpu())
    rank_relation_loss = 0
    position_relation_loss = 0
    shape_loss = 0

    if preserve_rank:
        rank_relation_loss = cal_rank_relation_loss(rep_embeddings, pre_rep_embeddings, cluster_indices,
                                                    exclude_indices, pre_pairwise_dist)

    if preserve_positions:
        position_relation_loss = cal_position_relation_loss(cluster_center_embeddings, pre_cluster_center_embeddings)

    if preserve_shape:
        pass

    return rank_weights * rank_relation_loss + position_weight + position_relation_loss + shape_weight + shape_loss


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
    return torch.mean(inner_pearson_corr) + torch.mean(intra_spearman_corr)
    # return torch.mean(inner_pearson_corr)


def _with_pairwise_dist_change(cur_embeddings, pre_embeddings, cluster_indices,
                               exclude_indices, pre_pairwise_dists=None):
    cluster_num = len(cluster_indices)
    inner_dist_change = torch.zeros(cluster_num)
    # intra_dist_change = torch.zeros(cluster_num)
    intra_spearman_corr = torch.zeros(cluster_num)

    cur_pairwise_dists = torch.cdist(cur_embeddings, cur_embeddings)
    pre_pairwise_dists = torch.cdist(pre_embeddings,
                                     pre_embeddings) if pre_pairwise_dists is None else pre_pairwise_dists
    for i, item in enumerate(cluster_indices):
        other_cluster_indices = exclude_indices[i]
        '''
        
        负例采样策略对聚类的精度还算是比较鲁棒的，即使将多个聚类分为一个类了，影响也还是比较小。
        秩序关系保持对聚类的精度也比较鲁棒。
        相对位置保持对聚类的精度也比较鲁棒。
        形状保持对聚类的精度也比较鲁棒。
        
        就是相对大小的保持受聚类精度的影响比较大。是否需要突出相对大小这一点？其实对局部结构的保持就能够看作是保持相对大小了。
        
        TODO：如果聚类不准确，这一项将会导致不同的聚类耦合在一起。直接约束邻居点对距离的变化比较合理。那么这一项就可以单独的看作LwF损失
        同时约束模型更新过程中局部结构的变化。然后在此基础上再加上全局模式变化的约束。
        
        与之前LwF方法的区别可以从目标这一点说，之前的分类任务可以保持输出不变，但是在降维场景中目标是保持局部结构不变。同时还要适应一部分数据的局部结构变化。
        
        从各聚类中随机采样一些点，以及这些点的邻居。根据权重约束更新前后的点对距离变化，作为局部结构（相对大小）保持损失。
        然后再基于各个聚类中的点，分别计算秩序关系保持损失、相对位置保持损失（计算在各个方向上的带方向距离的皮尔逊系数）、形状保持损失（聚类边缘点的相对位置关系）。
        '''
        inner_dist_change[i] = torch.mean(torch.norm(cur_pairwise_dists[item][:, item] -
                                                     pre_pairwise_dists[item][:, item], dim=-1))

        # intra_dist_change[i] = torch.mean(torch.norm(cur_pairwise_dists[item][:, other_cluster_indices] -
        #                                              pre_pairwise_dists[item][:, other_cluster_indices], dim=-1))
        idx = random.randint(0, len(item))
        intra_spearman_corr[i] = compute_rank_correlation(cur_pairwise_dists[item][:, other_cluster_indices],
                                                          pre_pairwise_dists[item][:, other_cluster_indices])

    # 约束inner_dist_change的变化可以看作是用于增量学习。但是仅依靠这一点无法保持视觉一致性，所以我们加上了intra_spearman_correlation。
    return torch.mean(inner_dist_change) - torch.mean(intra_spearman_corr)
    # return torch.mean(inner_dist_change)


def cal_lwf_loss(rep_embeddings, pre_rep_embeddings, pre_rep_neighbors_embeddings):
    dists = torch.norm(rep_embeddings - pre_rep_neighbors_embeddings, dim=-1)
    pre_dists = torch.norm(pre_rep_embeddings - pre_rep_neighbors_embeddings, dim=-1)
    dist_change = torch.mean(torch.square(dists - pre_dists))
    return dist_change


def cal_rank_relation_loss(rep_embeddings, pre_rep_embeddings, cluster_indices, exclude_indices,
                           pre_pairwise_dists=None, single=False):
    cluster_num = len(cluster_indices)
    intra_spearman_corr = torch.zeros(cluster_num)

    cur_pairwise_dists = torch.cdist(rep_embeddings, rep_embeddings)
    pre_pairwise_dists = torch.cdist(pre_rep_embeddings,
                                     pre_rep_embeddings) if pre_pairwise_dists is None else pre_pairwise_dists
    for i, item in enumerate(cluster_indices):
        if single:
            item = item[random.randint(0, len(item))]

        other_cluster_indices = exclude_indices[i]

        intra_spearman_corr[i] = compute_rank_correlation(cur_pairwise_dists[item][:, other_cluster_indices],
                                                          pre_pairwise_dists[item][:, other_cluster_indices])

    return -torch.mean(intra_spearman_corr)


def cal_position_relation_loss(cluster_center_embeddings: torch.Tensor,
                               pre_cluster_center_embeddings: torch.Tensor):
    cluster_nums = cluster_center_embeddings.shape[0]
    latent_dims = cluster_center_embeddings.shape[1]
    direction_dist = cluster_center_embeddings.unsqueeze(0).repeat(cluster_nums, 1, 1) - \
                     cluster_center_embeddings.unsqueeze(1).repeat(0, cluster_nums, 1)
    # cluster_nums * cluster_nums * 2
    pre_direction_dist = pre_cluster_center_embeddings.unsqueeze(0).repeat(cluster_nums, 1, 1) - \
                         pre_cluster_center_embeddings.unsqueeze(1).repeat(0, cluster_nums, 1)

    loss = 0
    for i in range(latent_dims):
        loss += -torch.mean(pearsonr(direction_dist[:, :, i], pre_direction_dist[:, :, i])[:, 0])

    return loss / latent_dims

