#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

import h5py
import numpy as np
from scipy.spatial.distance import cdist

from dataset.warppers import StreamingDatasetWrapper
from model.dr_models.ModelSets import MODELS
from model.scdr.dependencies.experiment import position_vis
from model.scdr.dependencies.scdr_utils import KeyPointsGenerater
from utils.common_utils import get_config, time_stamp_to_date_time_adjoin
from utils.metrics_tool import Metric, knn_score

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
import torch
from utils.constant_pool import *
import argparse
import time
from model.scdr.dependencies.cdr_experiment import CDRsExperiments
from model.scdr.model_trainer import IncrementalCDREx

device = "cuda:0"
log_path = "logs/log.txt"


def query_knn(query_data, data_set, k):
    dists = cdist(query_data, data_set)
    knn_indices = np.argsort(dists, axis=1)[:, 1:k + 1]

    knn_distances = []
    for i in range(knn_indices.shape[0]):
        knn_distances.append(dists[i, knn_indices[i]])
    knn_distances = np.array(knn_distances)
    return knn_indices, knn_distances


class IndicesGenerator:
    def __init__(self, total_labels):
        self.total_labels = total_labels
        self.cls, self.cls_counts = np.unique(total_labels, return_counts=True)

    # 将指定标签的数据分成n份，每份的数据分布情况相同
    def no_dis_change(self, cls_num, n):
        batch_indices = [[] for i in range(n)]
        num_per_batch = self.cls_counts / n

        for i in range(n):
            for j in range(cls_num):
                cur_num = int(num_per_batch[j])
                indices = np.argwhere(self.total_labels == self.cls[j]).squeeze()
                batch_indices[i].extend(indices[i * cur_num:min((i + 1) * cur_num, len(indices))])

        return batch_indices

    # 将指定标签的数据分成n份，初始数据只来自其中50%的类别，之后的每份中数据均来自其他的一个类别
    def slightly_dis_change(self, total_cls_num, initial_cls_num):
        assert total_cls_num > initial_cls_num
        initial_batch_indices = []
        for i in range(initial_cls_num):
            initial_batch_indices.extend(np.argwhere(self.total_labels == self.cls[i]).squeeze())

        batch_indices = [initial_batch_indices]
        for i in range(initial_cls_num, total_cls_num):
            batch_indices.append(np.argwhere(self.total_labels == self.cls[i]).squeeze())

        return batch_indices

    # 初始数据只来自其中30%的类别，之后的每份中数据来自其他的多个类别
    def heavy_dis_change(self, initial_cls_num, after_cls_num_list):
        # 类别至少是 2 2 3的分布
        assert len(self.cls) >= initial_cls_num + np.sum(after_cls_num_list)

        initial_batch_indices = []
        for i in range(initial_cls_num):
            initial_batch_indices.extend(np.argwhere(self.total_labels == self.cls[i]).squeeze())

        batch_indices = [initial_batch_indices]
        pre = initial_cls_num
        for item in after_cls_num_list:
            tmp = []
            for i in range(pre, pre + item):
                tmp.extend(np.argwhere(self.total_labels == self.cls[i]).squeeze())
            batch_indices.append(tmp)
            pre += item
        return batch_indices


def incremental_cdr_pipeline():
    cfg.merge_from_file(CONFIG_PATH)
    result_save_dir = ConfigInfo.RESULT_SAVE_DIR.format(METHOD_NAME, cfg.method_params.n_neighbors,
                                                        cfg.exp_params.latent_dim)
    clr_model = MODELS[METHOD_NAME](cfg, device=device)
    experimenter = IncrementalCDREx(clr_model, cfg.exp_params.dataset, cfg, result_save_dir, CONFIG_PATH, shuffle=True,
                                    device=device, log_path=log_path)

    with h5py.File(os.path.join(ConfigInfo.DATASET_CACHE_DIR, "{}.h5".format(experimenter.dataset_name)), "r") as hf:
        total_data = np.array(hf['x'])
        total_labels = np.array(hf['y'])

    METRICS = ["Movement", "Neighbor Hit", "kNN-CA"]
    # 0.需要创建一个索引生成器，根据不同的条件生成一批批的索引列表
    indices_generator = IndicesGenerator(total_labels)

    # n_step, cls_num = 4, 5
    # batch_indices = indices_generator.no_dis_change(cls_num, n_step)

    # total_cls_num, initial_cls_num = 6, 3
    # n_steps = total_cls_num - initial_cls_num + 1
    # batch_indices = indices_generator.slightly_dis_change(total_cls_num=6, initial_cls_num=3)

    initial_cls_num, after_cls_num_list = 3, [2, 2, 3]
    n_step = len(after_cls_num_list) + 1
    batch_indices = indices_generator.heavy_dis_change(initial_cls_num, after_cls_num_list)

    embedding_acc_history = np.zeros(shape=(n_step, n_step, len(METRICS)))

    # 1.需要创建一个数据整合器，用于从旧数据中采样代表点数据
    initial_data = total_data[batch_indices[0]]
    initial_labels = total_labels[batch_indices[0]]
    stream_dataset = StreamingDatasetWrapper(initial_data, initial_labels, experimenter.batch_size)
    initial_embeddings = experimenter.first_train(stream_dataset, INITIAL_EPOCHS)
    position_vis(stream_dataset.total_label, os.path.join(RESULT_SAVE_DIR, "0.jpg"), initial_embeddings, title="Initial Embeddings")

    metric_tool = Metric(experimenter.dataset_name, stream_dataset.total_data, stream_dataset.total_label,
                         stream_dataset.knn_indices, stream_dataset.knn_distances, None, norm=experimenter.is_image,
                         k=experimenter.fixed_k)
    eval_res = evaluate(experimenter, metric_tool, initial_embeddings, initial_labels)
    embedding_acc_history[0, 0] = eval_res
    initial_embeddings_list = [initial_embeddings]

    fitted_indices = batch_indices[0]
    experimenter.active_incremental_learning()

    for i in range(1, n_step):
        # 之后应该要动态管理，固定代表性数据的规模，然后根据聚类数目动态分配
        rep_old_data, rep_old_indices = KeyPointsGenerater.generate(total_data[fitted_indices], 0, min_num=100)

        # 更新数据集状态
        cur_batch_data = total_data[batch_indices[i]]
        cur_batch_num = len(batch_indices[i])
        fitted_num = len(fitted_indices)

        knn_indices, knn_dists = query_knn(cur_batch_data, np.concatenate([stream_dataset.total_data, cur_batch_data],
                                                                          axis=0), k=K)

        stream_dataset.add_new_data(cur_batch_data, knn_indices, knn_dists, total_labels[batch_indices[i]])

        stream_dataset.update_knn_graph(total_data[fitted_indices], total_data[batch_indices[i]],
                                        [0, cur_batch_num])
        experimenter.update_batch_size(cur_batch_num)
        experimenter.update_dataloader(RESUME_EPOCH, np.arange(fitted_num, fitted_num + cur_batch_num, 1))

        # 2.将代表点数据作为候选负例兼容到模型的增量训练中，每次都需要计算代表性数据的嵌入
        rep_old_embeddings = experimenter.acquire_latent_code_allin(torch.tensor(rep_old_data, dtype=torch.float)
                                                                    .to(device), device)

        # 3.需要改变模型的损失计算。使用不同的方式进行incremental learning。需要创建一个新的train方法。
        embeddings = experimenter.resume_train(RESUME_EPOCH, rep_old_data, rep_old_embeddings)
        position_vis(stream_dataset.total_label, os.path.join(RESULT_SAVE_DIR, "{}.jpg".format(i)), embeddings,
                     title="Step {} Embeddings".format(i))

        # 4.需要比较新的模型对之前数据的投影质量，并且评估对新数据的投影质量
        metric_tool = Metric(experimenter.dataset_name, stream_dataset.total_data, stream_dataset.total_label,
                             None, None, None, norm=experimenter.is_image, k=experimenter.fixed_k)

        initial_embeddings_list.append(embeddings[-cur_batch_num:])
        tmp = 0
        for j in range(i+1):
            eval_indices = np.arange(tmp, tmp+len(batch_indices[j]), 1)
            tmp += len(batch_indices[j])

            pre_old_data_embeddings = initial_embeddings_list[j]
            # 新数据的加入可能导致旧数据的kNN发生变化，那么其嵌入标准实际上也发生了改变。这种情况下如何比较投影质量的变化呢？
            # 衍生出的另一个问题就是，被选作代表点的数据，应该是kNN没有发生变化或者变化非常小的数据。
            eval_res = evaluate(experimenter, metric_tool, embeddings, stream_dataset.total_label,
                                eval_indices, pre_old_data_embeddings)
            embedding_acc_history[j, i] = eval_res

        # 上述步骤1-4应该写成一个循环
        fitted_indices = np.concatenate([fitted_indices, batch_indices[i]])

    for i in range(n_step):
        print("{}th Data:".format(i))
        for j in range(len(METRICS)):
            print("-----{}:".format(METRICS[j]), embedding_acc_history[i, :, j])
        print()


def evaluate(experimenter, metric_tool, embeddings, labels, eval_indices=None, pre_embeddings=None):
    if pre_embeddings is not None:
        assert eval_indices is not None
        movements = np.mean(np.linalg.norm(embeddings[eval_indices] - pre_embeddings, axis=1))
    else:
        movements = 0

    neighbor_hit = metric_tool.metric_neighborhood_hit(experimenter.fixed_k, embeddings, eval_indices)
    knn_ca = knn_score(embeddings, labels, k=10, knn_indices=metric_tool.low_knn_indices, eval_indices=eval_indices)
    return movements, neighbor_hit, knn_ca


if __name__ == '__main__':
    with torch.cuda.device(0):
        time_step = time_stamp_to_date_time_adjoin(time.time())
        RESULT_SAVE_DIR = r"results/incremental ex/{}".format(time_step)
        os.mkdir(RESULT_SAVE_DIR)
        METHOD_NAME = "LwF_CDR"
        CONFIG_PATH = "configs/ParallelSCDR.yaml"
        INDICES_DIR = r"../../Data/indices/single_cluster"
        INITIAL_EPOCHS = 200
        RESUME_EPOCH = 50
        K = 10

        cfg = get_config()
        cfg.merge_from_file(CONFIG_PATH)
        indices_path = os.path.join(INDICES_DIR, "{}.npy".format(cfg.exp_params.dataset))

        incremental_cdr_pipeline()
