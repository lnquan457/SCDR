import math

from autograd import grad
import autograd.numpy as np
from autograd import elementwise_grad

from embedding_main import mae_loss_gard
from utils.loss_grads import norm_grad, umap_sim_grad
from utils.umap_utils import convert_distance_to_probability, find_ab_params


def norm_cal(x1, x2):
    y = (np.sum((x1 - x2)**2))**0.5
    return y


def to_umap(cur_dist):
    return 1.0 / (1.0 + a * cur_dist ** (2 * b))


def mae_loss(center_embedding, neighbors_embeddings, high_sims):
    cur_dist = norm_cal(center_embedding, neighbors_embeddings)

    low_sims = to_umap(cur_dist)
    loss = abs(low_sims - high_sims)
    return loss


if __name__ == '__main__':
    x1 = np.array([-0.21837, 0.32073])
    x2 = np.array([1.61784, 0.16449])
    y = 0.9458091
    a, b = find_ab_params(1.0, 0.1)

    mae_grads = elementwise_grad(mae_loss)

    dx = mae_grads(x1, x2, y)

    cccdist = norm_cal(x1, x2)
    print("ccc dist", cccdist)
    lll_sims = 1.0 / (1.0 + a * cccdist ** (2 * b))
    # my_dx = mae_loss_gard(x1, np.array([x2]), np.array([y]), np.array([cccdist]), np.array([lll_sims]), a, b)
    # print(my_dx)
    # print(dx)

    # 我们计算的大4倍
    norm_grads_func = elementwise_grad(norm_cal)
    dx_norm = norm_grads_func(x1, x2)
    my_dx_norm = norm_grad(x1, x2, cccdist)
    print(dx_norm)
    print(my_dx_norm)

    # 对了
    umap_grads_func = elementwise_grad(to_umap)
    dx_umap = umap_grads_func(cccdist)
    my_dx_umap = umap_sim_grad(cccdist, a, b)
    print(dx_umap)
    print(my_dx_umap)

