import math
import time

from model.dr_models.CDRs.NCELoss import NT_Xent, torch_app_skewnorm_func, Mixture_NT_Xent
from model.dr_models.CDRs.nx_cdr import NxCDRModel
import torch

from model.dr_models.CDRs.vc_loss import temporal_steady_loss, cal_space_expand_loss


class CDRModel(NxCDRModel):
    def __init__(self, cfg, device='cuda'):
        NxCDRModel.__init__(self, cfg, device)
        self.ratio = math.exp(1 / self.temperature) / torch.max(torch_app_skewnorm_func(torch.linspace(0, 1, 1000), 1))

        self.a = torch.tensor(-40)
        self.loc = torch.tensor(cfg.method_params.split_upper)
        self.lower_thresh = torch.tensor(cfg.method_params.split_lower)
        self.scale = torch.tensor(0.13)
        self.alpha = torch.tensor(cfg.method_params.alpha)
        self.cfg = cfg
        self.separate_epoch = int(self.epoch_num * cfg.method_params.separation_begin_ratio)
        self.steady_epoch = int(self.epoch_num * cfg.method_params.steady_begin_ratio)
        self.pre_epochs = self.epoch_num

    def update_separate_period(self, epoch_num):
        self.separate_epoch = int(epoch_num * self.cfg.method_params.separation_begin_ratio) + self.pre_epochs
        self.steady_epoch = int(epoch_num * self.cfg.method_params.steady_begin_ratio) + self.pre_epochs
        self.pre_epochs += epoch_num

    def preprocess(self):
        self.build_model()
        self.criterion = NT_Xent.apply

    def _post_loss(self, logits, x_embeddings, epoch, *args):
        # 分别计算 正例关键损失 和 负例关键损失
        if self.separate_epoch <= epoch <= self.steady_epoch:
            epoch_ratio = torch.tensor((epoch - self.separate_epoch) / (self.steady_epoch - self.separate_epoch))
            cur_lower_thresh = 0.001 + (self.lower_thresh - 0.001) * epoch_ratio
            # cur_lower_thresh = torch.tensor(0)
            loss = Mixture_NT_Xent.apply(logits, self.temperature, self.alpha, self.a, self.loc,
                                         cur_lower_thresh, self.scale)
        else:
            loss = self.criterion(logits, self.temperature)

        return loss


class LwFCDR(CDRModel):

    def __init__(self, cfg, device='cuda', neg_num=None):
        CDRModel.__init__(self, cfg, device)
        self.neg_num = None
        self._cluster_repel_alpha = torch.tensor(6.0)
        # self._min_similarity = 0.001
        self.update_neg_num(neg_num)
        self.__novel_nce_weight = 1.0
        # 新数据与旧数据之间进行排斥的对比损失的权重
        self.__cluster_repel_weight = 1.0
        # 保持旧数据局部结构不变的权重
        self.__lwf_weight = 10.0
        self.__temporal_steady_weight = 1.0
        self.__expand_weight = 5.0

        # 用于计算VC损失
        self._rep_cluster_indices = None
        self._rep_exclude_indices = None

    def update_neg_num(self, new_neg_num):
        if self.neg_num is not None:
            self.neg_num = min(new_neg_num, self.batch_size)
            self.correlated_mask = (1 - torch.eye(self.neg_num)).type(torch.bool)

    def update_rep_data_info(self, cluster_indices, exclude_indices):
        self._rep_cluster_indices = cluster_indices
        self._rep_exclude_indices = exclude_indices

    def compute_loss(self, x_embeddings, x_sim_embeddings, *args):
        epoch = args[0]
        is_incremental_learning = args[1]
        novel_logits = self.batch_logits(x_embeddings, x_sim_embeddings, *args)
        novel_nce_loss = self._post_loss(novel_logits, None, epoch, None, *args)

        if not is_incremental_learning:
            return novel_nce_loss

        rep_embeddings, pre_rep_embeddings = args[2], args[3]
        pre_rep_neighbors_embeddings, rep_neighbor_embeddings = args[4], args[5]
        no_change_indices = args[6]
        steady_weights, neighbor_steady_weights = args[7], args[8]
        batch_idx = args[9]

        old_logits = self.cal_old_logits(x_embeddings, x_sim_embeddings, rep_embeddings, novel_logits)
        # 增加t值虽然可以增加低相似度负例的排斥力度，但是会导致困难负例很难被排斥开。
        cluster_repel_loss = self._post_loss(old_logits, None, epoch, None, *args)
        # cluster_repel_loss = self._cluster_repel_loss(old_logits, epoch)

        cluster_indices = self._rep_cluster_indices[batch_idx]
        exclude_indices = self._rep_exclude_indices[batch_idx]

        if pre_rep_neighbors_embeddings is None:
            lwf_loss = 0
        else:
            lwf_loss = self._cal_lwf_loss(rep_embeddings[no_change_indices], rep_neighbor_embeddings,
                                          pre_rep_embeddings[no_change_indices], pre_rep_neighbors_embeddings,
                                          steady_weights, neighbor_steady_weights)

        sta = time.time()
        # TODO: ts_loss的梯度计算和传播较为耗时！

        pairwise_dists = torch.cdist(rep_embeddings, rep_embeddings).cpu()
        pre_pairwise_dists = torch.cdist(pre_rep_embeddings, pre_rep_embeddings).cpu()

        ts_loss, rank_loss, position_loss, shape_loss = \
            temporal_steady_loss(preserve_rank=True, preserve_positions=True, preserve_shape=True,
                                 rank_weights=1.0, position_weight=2.0, shape_weight=2.0,
                                 rep_embeddings=rep_embeddings, pre_rep_embeddings=pre_rep_embeddings,
                                 rep_neighbors_embeddings=rep_neighbor_embeddings,
                                 pre_rep_neighbors_embeddings=pre_rep_neighbors_embeddings,
                                 no_change_indices=no_change_indices,
                                 cluster_indices=cluster_indices, exclude_indices=exclude_indices,
                                 steady_weights=steady_weights, neighbor_steady_weights=neighbor_steady_weights,
                                 pairwise_dist=pairwise_dists, pre_pairwise_dist=pre_pairwise_dists)
        # print("temporal steady loss:", time.time() - sta)

        expand_loss = cal_space_expand_loss(rep_embeddings, pre_rep_embeddings, cluster_indices, exclude_indices,
                                            pairwise_dists, pre_pairwise_dists)

        # print("novel nce loss:", novel_nce_loss.item())
        # print("cluster repel loss:", cluster_repel_loss.item())
        # print("lwf loss:", lwf_loss.item())
        # print("temporal steady loss:", ts_loss.item())
        # print()

        w_novel_nce_loss = self.__novel_nce_weight * novel_nce_loss
        w_cluster_repel_loss = self.__cluster_repel_weight * cluster_repel_loss
        w_lwf_loss = self.__lwf_weight * lwf_loss
        w_ts_loss = self.__temporal_steady_weight * ts_loss
        w_expand_loss = self.__expand_weight * expand_loss

        loss = w_novel_nce_loss + w_cluster_repel_loss + w_lwf_loss + w_ts_loss + w_expand_loss
        return loss, w_novel_nce_loss, w_cluster_repel_loss, w_lwf_loss, w_expand_loss, rank_loss, position_loss, shape_loss

    def cal_old_logits(self, x_embeddings, x_sim_embeddings, rep_old_embeddings, novel_logits):
        pos_similarities = torch.clone(novel_logits[:, 0]).unsqueeze(1)

        rep_old_embeddings_matrix = rep_old_embeddings.unsqueeze(0).repeat(x_embeddings.shape[0] * 2, 1, 1)
        x_and_x_sim_embeddings = torch.cat([x_embeddings, x_sim_embeddings], dim=0)
        x_embeddings_matrix = x_and_x_sim_embeddings.unsqueeze(1).repeat(1, rep_old_embeddings.shape[0], 1)
        rep_old_negatives = self.similarity_func(rep_old_embeddings_matrix, x_embeddings_matrix, self.min_dist)[0]
        # print("neg num:", rep_old_negatives.shape[1])

        old_logits = torch.cat([pos_similarities, rep_old_negatives], dim=1)
        return old_logits

    def batch_logits(self, x_embeddings, x_sim_embeddings, *args):
        is_incremental_learning = args[1]
        if not is_incremental_learning or self.neg_num is None or self.neg_num == 2 * self.batch_size:
            logits = super().batch_logits(x_embeddings, x_sim_embeddings, *args)
        else:
            cur_batch_size = x_embeddings.shape[0]
            cur_available_neg_num = 2 * cur_batch_size - 2

            if cur_available_neg_num < self.neg_num:
                neg_num = cur_available_neg_num
                if self.partial_corr_mask is None:
                    self.partial_corr_mask = (1 - torch.eye(neg_num)).type(torch.bool)

                corr_mask = self.partial_corr_mask
            else:
                neg_num = self.neg_num
                corr_mask = self.correlated_mask

            queries = torch.cat([x_embeddings, x_sim_embeddings], dim=0)
            queries_matrix = queries[:neg_num + 1].unsqueeze(0).repeat(2 * cur_batch_size, 1, 1)
            positives = torch.cat([x_sim_embeddings, x_embeddings], dim=0)
            negatives_p1 = torch.cat([queries_matrix[:neg_num, :neg_num][corr_mask]
                                     .view(neg_num, neg_num - 1, -1),
                                      queries_matrix[:neg_num, neg_num].unsqueeze(1)], dim=1)

            negatives_p2 = torch.cat([queries_matrix[cur_batch_size:cur_batch_size + neg_num, :neg_num]
                                      [corr_mask].view(neg_num, neg_num - 1, -1),
                                      queries_matrix[cur_batch_size:cur_batch_size + neg_num, neg_num]
                                     .unsqueeze(1)], dim=1)

            negatives = torch.cat([negatives_p1, queries_matrix[neg_num:cur_batch_size, :neg_num],
                                   negatives_p2, queries_matrix[cur_batch_size + neg_num:, :neg_num]], dim=0)

            pos_similarities = self.similarity_func(queries, positives, self.min_dist)[0].unsqueeze(1)
            neg_similarities = self.similarity_func(queries_matrix[:, :neg_num], negatives, self.min_dist)[0]

            # 将本地的正例、负例与全局的负例拼接在一起
            logits = torch.cat((pos_similarities, neg_similarities), dim=1)
        return logits

    def _cal_lwf_loss(self, rep_embeddings, rep_neighbors_embeddings, pre_rep_embeddings, pre_rep_neighbors_embeddings,
                      steady_weights=None, neighbor_steady_weights=None):
        sim = self.similarity_func(rep_embeddings, rep_neighbors_embeddings)[0]
        # TODO: 这里的计算是有重复的，可以避免
        pre_sims = self.similarity_func(pre_rep_embeddings, pre_rep_neighbors_embeddings)[0]
        sim_change = sim - pre_sims
        if steady_weights is not None and neighbor_steady_weights is not None:
            sim_change *= (steady_weights + neighbor_steady_weights) / 2
        return torch.mean(torch.square(sim_change))

    def _cluster_repel_loss(self, logits, epoch):
        # logits[logits < self._min_similarity] = self._min_similarity
        if self.separate_epoch <= epoch <= self.steady_epoch:
            cur_lower_thresh = torch.tensor(0)
            loss = Mixture_NT_Xent.apply(logits, self.temperature, self._cluster_repel_alpha, self.a, self.loc,
                                         cur_lower_thresh, self.scale)
        else:
            loss = self.criterion(logits, self.temperature)

        return loss
