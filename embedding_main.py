import os
import time
import numpy as np
import h5py
import scipy
import torch
from scipy.spatial.distance import cdist

from dataset.warppers import StreamingDatasetWrapper
from model.dr_models.ModelSets import MODELS
from model.scdr.dependencies.experiment import position_vis
from model.scdr.model_trainer import IncrementalCDREx
from model_update_main import IndicesGenerator, query_knn
from utils.common_utils import time_stamp_to_date_time_adjoin, get_config
from utils.constant_pool import ConfigInfo
from utils.metrics_tool import Metric
from utils.umap_utils import find_ab_params, convert_distance_to_probability
from scipy import optimize

device = "cuda:0"


def trust(high_nn_indices, high_knn_indices, low_knn_indices, k):
    U = np.setdiff1d(low_knn_indices, high_knn_indices)
    sum_j = 0
    for j in range(U.shape[0]):
        high_rank = np.where(high_nn_indices == U[j])[0][0]
        sum_j += high_rank - k
    return 1 - (2 / (k * (2 - 3 * k - 1)) * sum_j)


def dists_loss(center_embedding, neighbor_embeddings):
    cur_dist = np.linalg.norm(center_embedding - neighbor_embeddings, axis=-1)
    return np.mean(cur_dist)


def umap_loss(center_embedding, neighbor_embeddings, high_sims, a, b):
    cur_dist = np.linalg.norm(center_embedding - neighbor_embeddings, axis=-1)
    low_sims = convert_distance_to_probability(cur_dist, a, b)
    loss = np.sum(-(high_sims * np.log(low_sims) + (1 - high_sims) * np.log(1 - low_sims)))
    return loss


def bfgs_optimize(initial_x, loss_func, neighbor_embeddings, high_sims, a, b):
    res = scipy.optimize.minimize(loss_func, initial_x, method="BFGS",
                                  args=(neighbor_embeddings, high_sims, a, b),
                                  options={'gtol': 1e-6, 'disp': False})
    return res


def incremental_cdr_pipeline():
    def _print_some(pre_text, nc_list, ar_list):
        mean_nc = np.mean(nc_list)
        std_nc = np.std(nc_list)
        mean_ar = np.mean(ar_list)
        std_ar = np.std(ar_list)
        print(pre_text, " Mean NC: %.2f Std NC: %.2f Mean AR: %.2f Std AR: %.2f" % (mean_nc, std_nc, mean_ar, std_ar))

    cfg.merge_from_file(CONFIG_PATH)
    clr_model = MODELS[METHOD_NAME](cfg, device=device)
    t_log_path = os.path.join(RESULT_SAVE_DIR, "logs.txt")
    experimenter = IncrementalCDREx(clr_model, cfg.exp_params.dataset, cfg, RESULT_SAVE_DIR, CONFIG_PATH, shuffle=True,
                                    device=device, log_path=t_log_path)
    n_neighbors = experimenter.n_neighbors
    a, b = find_ab_params(1.0, MIN_DIST)

    with h5py.File(os.path.join(ConfigInfo.DATASET_CACHE_DIR, "{}.h5".format(experimenter.dataset_name)), "r") as hf:
        total_data = np.array(hf['x'])
        total_labels = np.array(hf['y'], dtype=int)

    # 0.需要创建一个索引生成器，根据不同的条件生成一批批的索引列表
    indices_generator = IndicesGenerator(total_labels)

    # n_step, cls_num = 4, 5
    # batch_indices = indices_generator.no_dis_change(cls_num, n_step)

    total_manifold_num, initial_manifold_num = 6, 3
    n_step = total_manifold_num - initial_manifold_num + 1
    batch_indices = indices_generator.mixing(initial_manifold_num, total_manifold_num)

    # initial_cls_num, after_cls_num_list = 3, [2, 2, 3]
    # initial_cls_num, after_cls_num_list = 3, [1, 2, 3]
    # n_step = len(after_cls_num_list) + 1
    # batch_indices = indices_generator.heavy_dis_change(initial_cls_num, after_cls_num_list)

    # 1.需要创建一个数据整合器，用于从旧数据中采样代表点数据
    initial_data = total_data[batch_indices[0]]
    initial_labels = total_labels[batch_indices[0]]
    fitted_manifolds = set(np.unique(initial_labels))
    stream_dataset = StreamingDatasetWrapper(initial_data, initial_labels, experimenter.batch_size,
                                             experimenter.n_neighbors)
    initial_embeddings = experimenter.first_train(stream_dataset, INITIAL_EPOCHS)
    position_vis(stream_dataset.total_label, os.path.join(RESULT_SAVE_DIR, "0.jpg"), initial_embeddings,
                 title="Initial Embeddings")

    experimenter.active_incremental_learning()
    fitted_indices = batch_indices[0]
    total_neighbor_cover_list = [[], []]
    total_avg_rank_list = [[], []]
    before_embedding = []
    after_embedding = []
    changed_indices = []
    before_t_list = []
    after_t_list = []

    for i in range(1, n_step):
        cur_batch_data = total_data[batch_indices[i]]
        cur_batch_num = len(batch_indices[i])
        batch_neighbor_cover_list = [[], []]
        batch_avg_rank_list = [[], []]

        for j in range(cur_batch_num):
            cur_data_idx = batch_indices[i][j]
            cur_data = cur_batch_data[j][np.newaxis, :]
            cur_label = total_labels[cur_data_idx]
            cur_embedding = experimenter.cal_lower_embeddings(cur_data)[np.newaxis, :]

            knn_indices, knn_dists, sort_indices = \
                query_knn(cur_data, np.concatenate([stream_dataset.total_data, cur_data], axis=0), k=K,
                          return_indices=True)
            low_knn_indices, low_knn_dists = query_knn(cur_embedding, np.concatenate([experimenter.pre_embeddings,
                                                                                      cur_embedding], axis=0), k=K)
            before_mean_dist = np.mean(low_knn_dists)

            neighbor_cover = len(np.intersect1d(knn_indices[0], low_knn_indices[0])) / n_neighbors
            # 比较简单高效的方法就是直接基于neighbor cover来判断，低于之前的均值-sigma就使用插值法来确定初始嵌入
            dis_change = int(cur_label) not in fitted_manifolds
            # dis_change = neighbor_cover < 0.21 - 0.14   # mean - alpha * std

            # dists2pre_embeddings = cdist(cur_embedding, experimenter.pre_embeddings)
            # sort_indices = np.argsort(dists2pre_embeddings).squeeze()
            # average_rank = 0
            # for item in knn_indices[0]:
            #     tmp_idx = int(np.argwhere(sort_indices == item).squeeze())
            #     average_rank += tmp_idx

            rec_idx = 1 if dis_change else 0
            # avg_rank = average_rank / n_neighbors
            batch_neighbor_cover_list[rec_idx].append(neighbor_cover)
            batch_avg_rank_list[rec_idx].append(0)

            stream_dataset.add_new_data(cur_data, knn_indices, knn_dists, cur_label)
            stream_dataset.update_knn_graph(total_data[fitted_indices], cur_data, [0], update_similarity=True)

            if not dis_change:
                final_embedding = cur_embedding
            else:
                changed_indices.append(len(fitted_indices))
                before_embedding.append(cur_embedding)

                cur_neighbor_embeddings = experimenter.pre_embeddings[knn_indices[0]]
                neighbor_sims = stream_dataset.raw_knn_weights[-1]
                normed_neighbor_sims = neighbor_sims / np.sum(neighbor_sims)
                # cur_initial_embedding = cur_embedding
                cur_initial_embedding = np.sum(normed_neighbor_sims[:, np.newaxis] * cur_neighbor_embeddings, axis=0)
                # 这里还是需要计算数据到高维邻居的相似度
                # res = bfgs_optimize(cur_initial_embedding, umap_loss, experimenter.pre_embeddings[knn_indices[0]],
                #                     neighbor_sims, a, b)
                res = scipy.optimize.minimize(dists_loss, cur_initial_embedding, method="BFGS",
                                              args=(experimenter.pre_embeddings[knn_indices[0]]),
                                              options={'gtol': 1e-6, 'disp': False})

                final_embedding = res.x[np.newaxis, :]
                after_embedding.append(final_embedding)

                after_low_knn_indices, after_low_knn_dists = \
                    query_knn(final_embedding, np.concatenate([experimenter.pre_embeddings, final_embedding], axis=0),
                              k=K)
                before_trust = trust(sort_indices, knn_indices, low_knn_indices, n_neighbors)
                after_trust = trust(sort_indices, knn_indices, after_low_knn_indices, n_neighbors)
                # print(before_trust, after_trust)

                before_t_list.append(before_trust)
                after_t_list.append(after_trust)
                after_mean_dist = np.mean(after_low_knn_dists)
                after_neighbor_cover = len(np.intersect1d(knn_indices[0], low_knn_indices[0])) / n_neighbors
                if after_neighbor_cover - neighbor_cover > 0:
                    print("improve:", after_neighbor_cover - neighbor_cover, before_mean_dist - after_mean_dist)

            fitted_indices.append(cur_data_idx)
            experimenter.pre_embeddings = np.concatenate([experimenter.pre_embeddings, final_embedding], axis=0)

        # _print_some("No Change!", batch_neighbor_cover_list[0], batch_avg_rank_list[0])
        # _print_some("Change!", batch_neighbor_cover_list[1], batch_avg_rank_list[1])

        total_neighbor_cover_list[0].extend(batch_neighbor_cover_list[0])
        total_avg_rank_list[0].extend(batch_avg_rank_list[0])
        total_neighbor_cover_list[1].extend(batch_neighbor_cover_list[1])
        total_avg_rank_list[1].extend(batch_avg_rank_list[1])
        position_vis(stream_dataset.total_label, os.path.join(RESULT_SAVE_DIR, "{}.jpg".format(i)),
                     experimenter.pre_embeddings, title="Step {} Embeddings".format(i))

    _print_some("No Change!", total_neighbor_cover_list[0], total_avg_rank_list[0])
    _print_some("Change!", total_neighbor_cover_list[1], total_avg_rank_list[1])
    # TODO：需要对比一下直接使用模型嵌入，和基于牛顿法优化的嵌入质量差异
    print("Before Trust: %.4f After Trust: %.4f" % (np.mean(before_t_list), np.mean(after_t_list)))


if __name__ == '__main__':
    with torch.cuda.device(0):
        time_step = time_stamp_to_date_time_adjoin(time.time())
        RESULT_SAVE_DIR = r"results/embedding_ex/{}".format(time_step)
        os.mkdir(RESULT_SAVE_DIR)
        METHOD_NAME = "LwF_CDR"
        CONFIG_PATH = "configs/ParallelSCDR.yaml"

        MIN_DIST = 0.1
        INITIAL_EPOCHS = 100
        RESUME_EPOCH = 50
        REP_NUM = 100
        BATCH_WHOLE = True
        K = 10

        cfg = get_config()
        cfg.merge_from_file(CONFIG_PATH)

        incremental_cdr_pipeline()
