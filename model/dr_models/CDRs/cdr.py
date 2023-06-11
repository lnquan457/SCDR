import math
import random
import time

from model.dr_models.CDRs.NCELoss import NT_Xent, torch_app_skewnorm_func, Mixture_NT_Xent
from model.dr_models.CDRs.nx_cdr import NxCDRModel
import torch

from model.dr_models.CDRs.vc_loss import temporal_steady_loss, cal_space_expand_loss, cal_rank_relation_loss, \
    cal_position_relation_loss, cal_shape_loss


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
        self.separation_begin_rate = cfg.method_params.separation_begin_ratio
        self.steady_begin_ratio = cfg.method_params.steady_begin_ratio
        self.separate_epoch = int(self.epoch_num * self.separation_begin_rate)
        self.steady_epoch = int(self.epoch_num * self.steady_begin_ratio)
        self.pre_epochs = self.epoch_num

    def update_separation_rate(self, sepa_rate, steady_rate):
        self.separation_begin_rate = sepa_rate
        self.steady_begin_ratio = steady_rate

    def update_separate_period(self, epoch_num):
        self.separate_epoch = int(epoch_num * self.separation_begin_rate) + self.pre_epochs
        self.steady_epoch = int(epoch_num * self.steady_begin_ratio) + self.pre_epochs
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
        self.update_neg_num(neg_num)
        self.__novel_nce_weight = 1.0
        # 新数据与旧数据之间进行排斥的对比损失的权重
        self.__cluster_repel_weight = 1.0
        # 保持旧数据局部结构不变的权重
        self.__lwf_weight = 10.0     # 10.0
        self.__temporal_steady_weight = 1.0     # 1.0

        # 用于计算VC损失
        self.preserve_rank = True
        self.preserve_pos = True
        self.preserve_shape = True
        self.rank_weight = 1.0
        self.pos_weight = 2.0   # normal 2.0
        self.shape_weight = 2.0     # normal 2.0
        self._rep_cluster_indices = None
        self._rep_exclude_indices = None

        self.novel_nce_time = 0
        self.repel_time = 0
        self.lwf_time = 0
        self.ts_time = 0

    def update_neg_num(self, new_neg_num):
        if self.neg_num is not None:
            self.neg_num = min(new_neg_num, self.batch_size)
            self.correlated_mask = (1 - torch.eye(self.neg_num)).type(torch.bool)

    def update_rep_data_info(self, cluster_indices):
        self._rep_cluster_indices = cluster_indices

    def compute_loss(self, x_embeddings, x_sim_embeddings, *args):
        epoch = args[0]
        is_incremental_learning = args[1]
        sta = time.time()
        novel_logits = self.batch_logits(x_embeddings, x_sim_embeddings, *args)
        novel_nce_loss = self._post_loss(novel_logits, None, epoch, None, *args)

        if not is_incremental_learning:
            return novel_nce_loss

        # print("===================================")
        self.novel_nce_time += time.time() - sta
        # print("novel nce:", time.time() - sta, self.novel_nce_time)

        rep_embeddings, pre_rep_embeddings = args[2], args[3]
        pre_rep_neighbors_embeddings, rep_neighbor_embeddings = args[4], args[5]
        no_change_indices = args[6]
        steady_weights, neighbor_steady_weights = args[7], args[8]
        batch_idx = args[9]

        sta = time.time()
        old_logits = self.cal_old_logits(x_embeddings, x_sim_embeddings, rep_embeddings, novel_logits)
        cluster_repel_loss = self._post_loss(old_logits, None, epoch, None, *args)
        self.repel_time += time.time() - sta
        # print("repel:", time.time() - sta, self.repel_time)

        cluster_indices = self._rep_cluster_indices[batch_idx]

        if pre_rep_neighbors_embeddings is None:
            lwf_loss = 0
        else:
            sta = time.time()
            lwf_loss = self._cal_lwf_loss(rep_embeddings[no_change_indices], rep_neighbor_embeddings,
                                          pre_rep_embeddings[no_change_indices], pre_rep_neighbors_embeddings,
                                          steady_weights, neighbor_steady_weights)
            self.lwf_time += time.time() - sta
            # print("lwf:", time.time() - sta, self.lwf_time)

        t_sta = time.time()

        rank_loss = torch.tensor(0)
        position_loss = torch.tensor(0)
        shape_loss = torch.tensor(0)
        cluster_num = cluster_indices.shape[0]
        c_indices = [item[random.randint(0, len(item) - 1)] for item in cluster_indices]

        # TODO: 排斥旧数据以及preserve rank和position的时候用到了聚类相关的信息，但是对于这些方法，聚类精度的影响并不是特别大。
        if self.preserve_rank and cluster_num > 1:
            sta = time.time()
            rank_loss = cal_rank_relation_loss(rep_embeddings[c_indices], pre_rep_embeddings[c_indices])
            # print("     rank", time.time() - sta)

        if self.preserve_pos and cluster_num > 1:
            sta = time.time()
            position_loss = cal_position_relation_loss(rep_embeddings[c_indices], pre_rep_embeddings[c_indices])
            # print("     positions", time.time() - sta)

        if self.preserve_shape and len(no_change_indices) > 0:
            sta = time.time()
            shape_loss = cal_shape_loss(rep_embeddings[no_change_indices], rep_neighbor_embeddings,
                                        pre_rep_embeddings[no_change_indices], pre_rep_neighbors_embeddings,
                                        steady_weights, neighbor_steady_weights)
            # print("     shape:", time.time() - sta)

        w_rank_relation_loss = self.rank_weight * rank_loss
        w_position_relation_loss = self.pos_weight * position_loss
        w_shape_loss = self.shape_weight * shape_loss

        ts_loss = w_rank_relation_loss + w_position_relation_loss + w_shape_loss
        self.ts_time += time.time() - t_sta
        # print("temporal steady loss:", time.time() - t_sta, self.ts_time)

        w_novel_nce_loss = self.__novel_nce_weight * novel_nce_loss
        w_cluster_repel_loss = self.__cluster_repel_weight * cluster_repel_loss
        w_lwf_loss = self.__lwf_weight * lwf_loss
        w_ts_loss = self.__temporal_steady_weight * ts_loss

        loss = w_novel_nce_loss + w_cluster_repel_loss + w_lwf_loss + w_ts_loss
        return loss, w_novel_nce_loss, w_cluster_repel_loss, w_lwf_loss, rank_loss, position_loss, shape_loss

    def cal_old_logits(self, x_embeddings, x_sim_embeddings, rep_old_embeddings, novel_logits):
        pos_similarities = torch.clone(novel_logits[:, 0]).unsqueeze(1)

        rep_old_embeddings_matrix = rep_old_embeddings.unsqueeze(0).repeat(x_embeddings.shape[0] * 2, 1, 1)
        x_and_x_sim_embeddings = torch.cat([x_embeddings, x_sim_embeddings], dim=0)
        x_embeddings_matrix = x_and_x_sim_embeddings.unsqueeze(1).repeat(1, rep_old_embeddings.shape[0], 1)
        rep_old_negatives = self.similarity_func(rep_old_embeddings_matrix, x_embeddings_matrix, self.min_dist)[0]

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
