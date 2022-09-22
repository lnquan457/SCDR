import math
from model.dr_models.CDRs.NCELoss import NT_Xent, torch_app_skewnorm_func, Mixture_NT_Xent
from model.dr_models.CDRs.nx_cdr import NxCDRModel
import torch


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
            loss = Mixture_NT_Xent.apply(logits, torch.tensor(self.temperature), self.alpha, self.a, self.loc,
                                         cur_lower_thresh, self.scale)
        else:
            loss = self.criterion(logits, torch.tensor(self.temperature))

        return loss


class LwFCDR(CDRModel):
    def batch_logits(self, x_embeddings, x_sim_embeddings, *args):
        is_incremental_learning = args[3]
        logits = super().batch_logits(x_embeddings, x_sim_embeddings, *args)
        if not is_incremental_learning:
            return logits
        rep_old_embeddings = args[1]

        rep_old_embeddings_matrix = rep_old_embeddings.unsqueeze(0).repeat(x_embeddings.shape[0] * 2, 1, 1)
        x_and_x_sim_embeddings = torch.cat([x_embeddings, x_sim_embeddings], dim=0)
        x_embeddings_matrix = x_and_x_sim_embeddings.unsqueeze(1).repeat(1, rep_old_embeddings.shape[0], 1)
        # TODO：如果代表性数据与新的数据有来自同一聚类的情况，那一直排斥他们是否会导致模型坍塌呢？
        rep_old_negatives = self.similarity_func(rep_old_embeddings_matrix, x_embeddings_matrix, self.min_dist)[0]

        total_logits = torch.cat([logits, rep_old_negatives], dim=1)
        return total_logits

    def compute_loss(self, x_embeddings, x_sim_embeddings, *args):
        contrastive_loss = super().compute_loss(x_embeddings, x_sim_embeddings, *args)
        is_incremental_learning = args[3]
        if not is_incremental_learning:
            return contrastive_loss

        cur_embeddings, pre_embeddings = args[1], args[2]
        normalization_loss = torch.mean(torch.norm(cur_embeddings - pre_embeddings, dim=1))
        # print("normalization_loss", normalization_loss)
        total_loss = contrastive_loss + normalization_loss
        return total_loss
