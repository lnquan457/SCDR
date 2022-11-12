import random
import time

import numpy as np
from torch.autograd import Function
import torch
import matplotlib.pyplot as plt
import math

normal_obj = torch.distributions.normal.Normal(0, 1)
EPS = 1e-7


def torch_norm_pdf(data):
    return torch.exp(-torch.square(data) / 2) / math.sqrt(2 * math.pi)


def torch_norm_cdf(data):
    global normal_obj
    return normal_obj.cdf(data)


def torch_skewnorm_pdf(data, a, loc, scale):
    # skewnorm_pdf(x, a, loc, scale) = skewnorm_pdf(y, a) / scale with y = (x - loc) / scale
    # skewnorm_pdf(x, a) = 2 * norm_pdf(x) * norm_cdf(a*x)
    # norm_pdf(x) = exp(-x^2/2) / sqrt(2*pi)
    # norm_cdf(x) = 1 - norm_pdf(x) / 0.7*x + sqrt(2/pi)
    y = (data - loc) / scale
    # output = 2 * norm_pdf(y) * norm_cdf(a*y) / scale
    output = 2 * torch_norm_pdf(y) * torch_norm_cdf(a * y) / scale
    return output


def torch_app_skewnorm_func(data, r, a=25, loc=0.05, scale=0.7):
    """

    :param data: 初始概率值
    :param r: 放缩比率
    :param a: 偏度，默认值为20。值越小则对简单负例的排斥越大，聚簇更加分离；值越大则对简单负例的排斥越小，聚簇越紧凑
    :param loc: 大致为峰值的位置，默认为0.05。与峰值大致成1.5倍的关系。
    :param scale: 尖锐性，默认为1.0。值越大函数越平缓，算法对困难负例的排斥力越大，簇内会更加发散；值越小函数越尖锐，算法对困难负例的排斥力越小，簇内会更加紧凑。
    :return: 放缩后的概率值
    """
    y = torch_skewnorm_pdf(data, a, loc, scale)
    y = y * r
    return y


class NT_Xent(Function):

    @staticmethod
    def forward(ctx, probabilities, t):
        exp_prob = torch.exp(probabilities / t)

        similarities = exp_prob / torch.sum(exp_prob, dim=1).unsqueeze(1)

        ctx.save_for_backward(similarities, t)

        pos_loss = -torch.log(similarities[:, 0]).mean()

        return pos_loss

    @staticmethod
    def backward(ctx, grad_output):
        # sta = time.time()
        similarities, t = ctx.saved_tensors

        # [2*B, 1]
        pos_grad_coeff = -((torch.sum(similarities, dim=1) - similarities[:, 0]) / t).unsqueeze(1)

        # [2*B, 2*(B-1)]
        # 在这里用一个平缓的分布进行变换，
        neg_grad_coeff = similarities[:, 1:] / t

        # [2*B, 2*B-1]
        grad = torch.cat([pos_grad_coeff, neg_grad_coeff], dim=1) * grad_output / similarities.shape[0]
        # print("nt xent:", time.time() - sta)
        return grad, None


class Mixture_NT_Xent(Function):

    @staticmethod
    def forward(ctx, probabilities, t, alpha, a, loc, lower_thresh, scale):

        similarities, exp_neg_grad_coeff = nt_xent_grad(probabilities, t)
        # ============================================负偏态分布计算============================================
        skewnorm_prob = torch_skewnorm_pdf(probabilities[:, 1:], a, loc, scale) + EPS
        skewnorm_similarities = skewnorm_prob / torch.sum(skewnorm_prob, dim=1).unsqueeze(1)
        sn_max_val_indices = torch.argmax(skewnorm_similarities, dim=1)
        rows = torch.arange(0, skewnorm_similarities.shape[0], 1)
        skewnorm_max_value = skewnorm_similarities[rows, sn_max_val_indices].unsqueeze(1)
        ref_exp_value = exp_neg_grad_coeff[rows, sn_max_val_indices].unsqueeze(1)

        raw_alpha = ref_exp_value / skewnorm_max_value

        ctx.save_for_backward(probabilities, similarities, t, skewnorm_similarities, loc, lower_thresh, alpha,
                              raw_alpha)
        # ============================================负偏态分布计算============================================

        pos_loss = -torch.log(similarities[:, 0]).mean()

        return pos_loss

    @staticmethod
    def backward(ctx, grad_output):
        # sta = time.time()
        prob, exp_sims, t, sn_sims, loc, lower_thresh, alpha, raw_alpha = ctx.saved_tensors

        # [2*B, 1]
        pos_grad_coeff = -((torch.sum(exp_sims, dim=1) - exp_sims[:, 0]) / t).unsqueeze(1)

        high_thresh = loc
        sn_sims[prob[:, 1:] < lower_thresh] = 0
        sn_sims[prob[:, 1:] >= high_thresh] = 0
        exp_sims[:, 1:][prob[:, 1:] < lower_thresh] = 0

        # [2*B, 2*(B-1)]
        # 在这里用一个平缓的分布进行变换，
        neg_grad_coeff = exp_sims[:, 1:] / t + alpha * sn_sims * raw_alpha

        # [2*B, 2*B-1]
        grad = torch.cat([pos_grad_coeff, neg_grad_coeff], dim=1) * grad_output / exp_sims.shape[0]
        # print("mixture:", time.time() - sta)
        return grad, None, None, None, None, None, None


def nt_xent_grad(data, tau):
    exp_prob = torch.exp(data / tau)
    norm_exp_prob = exp_prob / torch.sum(exp_prob, dim=1).unsqueeze(1)
    gradients = norm_exp_prob[:, 1:] / tau
    return norm_exp_prob, gradients
