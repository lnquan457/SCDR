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
