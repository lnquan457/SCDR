import math
from model.dr_models.CDRs.NCELoss import NT_Xent, torch_app_skewnorm_func, Mixture_NT_Xent
from model.dr_models.CDRs.nx_cdr import NxCDRModel
import torch

from model.dr_models.CDRs.vc_loss import _visual_consistency_loss


class CDRModel(NxCDRModel):
    def __init__(self, cfg, device='cuda'):
        NxCDRModel.__init__(self, cfg, device)
        self.ratio = math.exp(1 / self.temperature) / torch.max(torch_app_skewnorm_func(torch.linspace(0, 1, 1000), 1))

        self.a = torch.tensor(-40)
        # loc这个值是不是也可以动态改变，从大到小
        # 整个增强范围随着训练过程是有宽到窄的，而增强的强度则是由小到大的
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
            loss = Mixture_NT_Xent.apply(logits, torch.tensor(self.temperature), torch.tensor(self.alpha), self.a, self.loc,
                                         cur_lower_thresh, self.scale)
        else:
            loss = self.criterion(logits, torch.tensor(self.temperature))

        return loss


class LwFCDR(CDRModel):

    def __init__(self, cfg, device='cuda', neg_num=None):
        CDRModel.__init__(self, cfg, device)
        self.neg_num = None
        self.update_neg_num(neg_num)
        # 新数据与旧数据之间进行排斥的对比损失的权重
        self.alpha = 3.0
        # 保持旧数据嵌入位置不变的权重
        self.beta = 2.0

    def update_neg_num(self, new_neg_num):
        if self.neg_num is not None:
            self.neg_num = min(new_neg_num, self.batch_size)
            self.correlated_mask = (1 - torch.eye(self.neg_num)).type(torch.bool)

    def compute_loss(self, x_embeddings, x_sim_embeddings, *args):
        epoch = args[0]
        is_incremental_learning = args[1]
        novel_logits = self.batch_logits(x_embeddings, x_sim_embeddings, *args)
        novel_loss = self._post_loss(novel_logits, None, epoch, None, *args)

        if not is_incremental_learning:
            return novel_loss

        # self.alpha = 8

        rep_old_embeddings, pre_old_embeddings = args[2], args[3]
        cluster_indices, exclude_indices = args[4], args[5]
        pre_embeddings_pw_dists = args[6]
        # cluster_indices, exclude_indices = None, None

        old_logits = self.cal_old_logits(x_embeddings, x_sim_embeddings, rep_old_embeddings, novel_logits)
        # old_logits[:, 0] = old_logits[:, 0].detach()
        old_loss = self._post_loss(old_logits, None, epoch, None, *args)

        vc_loss = _visual_consistency_loss(rep_old_embeddings, pre_old_embeddings, cluster_indices=cluster_indices,
                                           exclude_indices=exclude_indices, pre_pairwise_dist=pre_embeddings_pw_dists)
        # print(" vc loss:", vc_loss.item())
        # print("novel loss:", novel_loss, " old loss:", old_loss, " vc loss:", vc_loss)
        loss = novel_loss + self.alpha * old_loss + self.beta * vc_loss
        return loss

    def cal_old_logits(self, x_embeddings, x_sim_embeddings, rep_old_embeddings, novel_logits):
        pos_similarities = torch.clone(novel_logits[:, 0]).unsqueeze(1)

        rep_old_embeddings_matrix = rep_old_embeddings.unsqueeze(0).repeat(x_embeddings.shape[0] * 2, 1, 1)
        x_and_x_sim_embeddings = torch.cat([x_embeddings, x_sim_embeddings], dim=0)
        x_embeddings_matrix = x_and_x_sim_embeddings.unsqueeze(1).repeat(1, rep_old_embeddings.shape[0], 1)
        # TODO：如果代表性数据与新的数据有来自同一聚类的情况，那一直排斥他们是否会导致模型坍塌呢？
        rep_old_negatives = self.similarity_func(rep_old_embeddings_matrix, x_embeddings_matrix, self.min_dist)[0]

        old_logits = torch.cat([pos_similarities, rep_old_negatives], dim=1)
        return old_logits

    def batch_logits(self, x_embeddings, x_sim_embeddings, *args):
        is_incremental_learning = args[1]
        if not is_incremental_learning or self.neg_num is None or self.neg_num == 2 * self.batch_size:
            logits = super().batch_logits(x_embeddings, x_sim_embeddings, *args)
        else:
            queries = torch.cat([x_embeddings, x_sim_embeddings], dim=0)
            queries_matrix = queries[:self.neg_num+1].unsqueeze(0).repeat(2*self.batch_size, 1, 1)
            positives = torch.cat([x_sim_embeddings, x_embeddings], dim=0)
            negatives_p1 = torch.cat([queries_matrix[:self.neg_num, :self.neg_num][self.correlated_mask]
                                     .view(self.neg_num, self.neg_num-1, -1),
                                      queries_matrix[:self.neg_num, self.neg_num].unsqueeze(1)], dim=1)

            negatives_p2 = torch.cat([queries_matrix[self.batch_size:self.batch_size+self.neg_num, :self.neg_num]
                                      [self.correlated_mask].view(self.neg_num, self.neg_num-1, -1),
                                      queries_matrix[self.batch_size:self.batch_size+self.neg_num, self.neg_num]
                                     .unsqueeze(1)], dim=1)

            negatives = torch.cat([negatives_p1, queries_matrix[self.neg_num:self.batch_size, :self.neg_num],
                                   negatives_p2, queries_matrix[self.batch_size+self.neg_num:, :self.neg_num]], dim=0)

            pos_similarities = self.similarity_func(queries, positives, self.min_dist)[0].unsqueeze(1)
            neg_similarities = self.similarity_func(queries_matrix[:, :self.neg_num], negatives, self.min_dist)[0]

            # 将本地的正例、负例与全局的负例拼接在一起
            logits = torch.cat((pos_similarities, neg_similarities), dim=1)
        return logits

