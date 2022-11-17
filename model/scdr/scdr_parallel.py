import os.path
import time

import numpy as np
import torch
from scipy.spatial.distance import cdist

from dataset.warppers import StreamingDatasetWrapper, DataRepo
from model.scdr.dependencies.embedding_optimizer import EmbeddingOptimizer
from utils.nn_utils import StreamingANNSearchKD, compute_knn_graph, StreamingANNSearchAnnoy
from utils.queue_set import ModelUpdateQueueSet, DataProcessorQueueSet
from model.scdr.dependencies.scdr_utils import KeyPointsGenerator, DistributionChangeDetector, ClusterRepDataSampler, \
    EmbeddingQualitySupervisor

OPTIMIZE_NEW_DATA_EMBEDDING = True
OPTIMIZE_NEIGHBORS = True


class SCDRParallel:
    def __init__(self, n_neighbors, batch_size, model_update_queue_set, initial_train_num, ckpt_path=None,
                 device="cuda:0"):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.model_update_queue_set = model_update_queue_set
        self.device = device
        self.infer_model = None
        self.ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
        self.initial_train_num = initial_train_num
        self.data_num_when_update = 0

        # 用于确定何时要更新模型，1）新的流形出现；2）模型对旧流形数据的嵌入质量太低；3）长时间没有更新；
        self.model_update_intervals = 600
        self.manifold_change_num_thresh = 300
        self.bad_embedding_num_thresh = 400

        # 是否进行跳步优化
        self.skip_opt = True
        # bfgs优化时,使用的负例数目
        self.opt_neg_num = 50

        # 模型优化时，每个batch中的旧数据采样率
        self.rep_data_sample_rate = 0.07
        self.rep_data_minimum_num = 50
        # 是否要采样所有旧数据
        self.cover_all = True

        # self.knn_searcher_approx = StreamingANNSearchKD()
        self.knn_searcher_approx = StreamingANNSearchAnnoy()

        self.cluster_rep_data_sampler = ClusterRepDataSampler(self.rep_data_sample_rate, self.rep_data_minimum_num,
                                                              self.cover_all)
        self.pre_cluster_centers = None
        # 进行初次训练后，初始化以下对象
        self.stream_dataset = StreamingDatasetWrapper(batch_size, n_neighbors)
        self.embedding_quality_supervisor = None
        self.super_acc = 0
        self.embedding_optimizer = None

        self.initial_data_buffer = None
        self.initial_label_buffer = None

        self.model_trained = False
        self.fitted_data_num = 0
        self.model_just_updated = False
        self._neighbor_changed_indices = set()
        self._neighbor_pos = np.arange(self.n_neighbors)

        self.knn_cal_time = 0
        self.knn_update_time = 0
        self.model_init_time = 0
        self.model_infer_time = 0
        self.embedding_opt_time = 0
        self.embedding_update_time = 0
        self.quality_record_time = 0

        self.debug = True
        self.update_when_end = False

    # scdr本身属于嵌入进程，负责数据的处理和嵌入
    def fit_new_data(self, data, labels=None):
        new_data_num = data.shape[0]
        pre_n_samples = self.stream_dataset.get_n_samples()

        if not self.model_trained:
            if not self._caching_initial_data(data, labels):
                return None

            self.stream_dataset.add_new_data(data=self.initial_data_buffer, labels=self.initial_label_buffer)
            cluster_indices = self._initial_project_model()
            self._initial_embedding_optimizer_and_quality_supervisor(cluster_indices)
        else:
            pre_embeddings = self.stream_dataset.get_total_embeddings()
            # 使用之前的投影函数对最新的数据进行投影
            sta = time.time()
            data_embeddings = self.infer_embeddings(data)
            self.model_infer_time += time.time() - sta

            sta = time.time()
            update = False
            if self.model_just_updated:
                update = True
                self.model_just_updated = False
            knn_indices, knn_dists = \
                self.knn_searcher_approx.search(self.n_neighbors, pre_embeddings,
                                                self.stream_dataset.get_total_data(), self.fitted_data_num,
                                                data_embeddings, data, update)
            # if np.max(knn_indices) >= pre_n_samples:
            #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!", knn_indices)
            knn_indices = knn_indices[np.newaxis, :]
            knn_dists = knn_dists[np.newaxis, :]
            self.knn_cal_time += time.time() - sta
            # 准确查询
            # acc_knn_indices, acc_knn_dists = query_knn(data, np.concatenate([self.stream_dataset.get_total_data(), data],
            #                                                                 axis=0), k=self.n_neighbors)

            self.stream_dataset.add_new_data(data, None, labels, knn_indices, knn_dists)

            sta = time.time()
            self.stream_dataset.update_knn_graph(pre_n_samples, data, [0], update_similarity=False,
                                                 symmetric=False)
            # if np.max(self.stream_dataset.get_knn_indices()) > pre_n_samples:
            #     print("==================================", knn_indices)
            self.knn_update_time += time.time() - sta

            # 两种邻居点策略：第一种是使用模型最新计算的，会增加时耗
            # 可以通过记录嵌入发生变化的数据，然后只对嵌入发生变化的进行重新嵌入，就可以减少嵌入的量。但对效率提升的作用不是很大。
            sta = time.time()

            neighbor_data = self.stream_dataset.get_total_data()[knn_indices.squeeze()]
            neighbor_embeddings = self.infer_embeddings(neighbor_data)
            self.model_infer_time += time.time() - sta

            sta = time.time()
            p_need_optimize, need_update_model = \
                self.embedding_quality_supervisor.quality_record(data, data_embeddings, self.pre_cluster_centers,
                                                                 neighbor_embeddings)
            self.quality_record_time += time.time() - sta

            # unique_labels, counts = np.unique(self.stream_dataset.get_total_label()[:self.fitted_data_num],
            #                                   return_counts=True)
            # tt_indices = np.where(counts > 30)[0]
            # need_optimize = labels not in (unique_labels[tt_indices])
            # if need_optimize == p_need_optimize:
            #     self.super_acc += 1
            # print(labels, unique_labels[tt_indices])
            # print(p_need_optimize, need_optimize, " supervise acc:", self.super_acc / (self.stream_dataset.get_n_samples() - self.initial_train_num))

            if need_update_model:
                self._send_update_signal()

            if not need_update_model and p_need_optimize:
                # ====================================1. 只对新数据本身的嵌入进行更新=======================================
                if OPTIMIZE_NEW_DATA_EMBEDDING:
                    sta = time.time()
                    data_embeddings = self.embedding_optimizer.optimize_new_data_embedding(
                        self.stream_dataset.raw_knn_weights[-1],
                        neighbor_embeddings, pre_embeddings)[np.newaxis, :]
                    self.embedding_opt_time += time.time() - sta
                # =====================================================================================================

            if OPTIMIZE_NEIGHBORS and not need_update_model:
                sta = time.time()
                neighbor_changed_indices, replaced_raw_weights, replaced_indices, anchor_positions = \
                    self.stream_dataset.get_pre_neighbor_changed_info()
                if len(neighbor_changed_indices) > 0:
                    self._neighbor_changed_indices = self._neighbor_changed_indices.union(neighbor_changed_indices)
                    optimized_embeddings = self.embedding_optimizer.update_old_data_embedding(
                        data_embeddings, pre_embeddings, neighbor_changed_indices,
                        self.stream_dataset.get_knn_indices(), self.stream_dataset.get_knn_dists(),
                        self.stream_dataset.raw_knn_weights[neighbor_changed_indices], anchor_positions,
                        replaced_indices, replaced_raw_weights)
                    self.embedding_update_time += time.time() - sta
                    self.stream_dataset.update_embeddings(optimized_embeddings)

            self.stream_dataset.add_new_data(embeddings=data_embeddings)

        return self.stream_dataset.get_total_embeddings()

    def _send_update_signal(self):

        acc_knn_indices, _ = compute_knn_graph(self.stream_dataset.get_total_data(), None, self.n_neighbors, None)
        acc = np.ravel(self.stream_dataset.get_knn_indices()) == np.ravel(acc_knn_indices)
        print("kNN acc:", np.sum(acc) / len(acc))

        sta = time.time()
        self.stream_dataset.update_cached_neighbor_similarities()
        self.knn_update_time += time.time() - sta
        # 如果还有模型更新任务没有完成，此处应该阻塞等待
        while self.model_update_queue_set.MODEL_UPDATING.value == 1:
            pass
        self.model_update_queue_set.MODEL_UPDATING.value = 1
        self.data_num_list = [0]
        self.data_num_when_update = self.stream_dataset.get_total_data().shape[0]
        self.model_update_queue_set.training_data_queue.put(
            [self.stream_dataset, self.cluster_rep_data_sampler, self.fitted_data_num,
             self.stream_dataset.get_n_samples() - self.fitted_data_num])
        self.model_update_queue_set.flag_queue.put(ModelUpdateQueueSet.UPDATE)

        # TODO: 使得模型更新和数据处理变成串行的
        while self.model_update_queue_set.MODEL_UPDATING.value:
            pass
        self.model_update_queue_set.WAITING_UPDATED_DATA.value = 1

    def get_final_embeddings(self):
        embeddings = self.stream_dataset.get_total_embeddings()
        if self.update_when_end:
            # print("final update")
            # 暂时先等待之前的模型训练完
            while self.model_update_queue_set.MODEL_UPDATING.value == 1:
                pass

            if not self.model_update_queue_set.embedding_queue.empty():
                self.model_update_queue_set.embedding_queue.get()  # 取出之前训练的结果，但是在这里是没用的了
            self._send_update_signal()
            embeddings, infer_model, stream_dataset, _ = self.model_update_queue_set.embedding_queue.get()
            self.update_scdr(infer_model, embeddings, stream_dataset)
        return embeddings

    def _caching_initial_data(self, data, labels):
        self.initial_data_buffer = data if self.initial_data_buffer is None \
            else np.concatenate([self.initial_data_buffer, data], axis=0)
        if labels is not None:
            self.initial_label_buffer = labels if self.initial_label_buffer is None \
                else np.concatenate([self.initial_label_buffer, labels], axis=0)

        return self.initial_data_buffer.shape[0] >= self.initial_train_num

    def _initial_project_model(self):
        # 这里传过去的stream_dataset中，包含了最新的数据、标签以及kNN信息
        self.data_num_when_update = self.stream_dataset.get_total_data().shape[0]
        self.model_update_queue_set.training_data_queue.put(
            [self.stream_dataset, self.cluster_rep_data_sampler, self.ckpt_path])
        self.model_update_queue_set.flag_queue.put(ModelUpdateQueueSet.UPDATE)
        self.model_update_queue_set.INITIALIZING.value = True
        # 这里传回来的stream_dataset中，包含了最新的对称kNN信息
        embeddings, model, stream_dataset, cluster_indices = self.model_update_queue_set.embedding_queue.get()
        self.update_scdr(model, embeddings, stream_dataset, first=True)
        self.stream_dataset.add_new_data(embeddings=embeddings)
        self.fitted_data_num = embeddings.shape[0]
        self.model_update_queue_set.INITIALIZING.value = False
        self.model_just_updated = True
        return cluster_indices

    def _initial_embedding_optimizer_and_quality_supervisor(self, cluster_indices):
        pre_neighbor_embedding_m_dist, pre_neighbor_embedding_s_dist, pre_neighbor_mean_dist, \
        pre_neighbor_std_dist, mean_dist2clusters, std_dist2clusters = self.get_pre_embedding_statistics(
            cluster_indices)

        # =====================================初始化embedding_quality_supervisor===================================

        d_thresh = mean_dist2clusters + 1 * std_dist2clusters
        e_thresh = pre_neighbor_embedding_m_dist + 1 * pre_neighbor_embedding_s_dist
        print("d thresh:", d_thresh, " e thresh:", e_thresh)
        self.embedding_quality_supervisor = EmbeddingQualitySupervisor(self.model_update_intervals,
                                                                       self.manifold_change_num_thresh,
                                                                       self.bad_embedding_num_thresh, d_thresh,
                                                                       e_thresh)
        self.embedding_quality_supervisor.update_model_update_time(time.time())
        # ==========================================================================================================

        # =====================================初始化embedding_optimizer==============================================
        local_move_thresh = pre_neighbor_embedding_m_dist + 3 * pre_neighbor_embedding_s_dist
        bfgs_update_thresh = pre_neighbor_mean_dist + 1 * pre_neighbor_std_dist
        self.embedding_optimizer = EmbeddingOptimizer(local_move_thresh, bfgs_update_thresh,
                                                      neg_num=self.opt_neg_num, skip_opt=self.skip_opt)
        # ===========================================================================================================

    def get_pre_embedding_statistics(self, cluster_indices=None):
        pre_neighbor_mean_dist, pre_neighbor_std_dist = self.stream_dataset.get_data_neighbor_mean_std_dist()
        pre_neighbor_embedding_m_dist, pre_neighbor_embedding_s_dist = \
            self.stream_dataset.get_embedding_neighbor_mean_std_dist()
        # print(pre_neighbor_mean_dist, pre_neighbor_std_dist)
        # print(pre_neighbor_embedding_m_dist, pre_neighbor_embedding_s_dist)
        if cluster_indices is None:
            cluster_indices = self.cluster_rep_data_sampler.sample(self.stream_dataset.get_total_embeddings(),
                                                                   pre_neighbor_embedding_m_dist, self.n_neighbors,
                                                                   self.stream_dataset.get_total_label())[-1]
        self.pre_cluster_centers, mean_d, std_d = \
            self.cluster_rep_data_sampler.dist_to_nearest_cluster_centroids(self.stream_dataset.get_total_data(),
                                                                            cluster_indices)
        return pre_neighbor_embedding_m_dist, pre_neighbor_embedding_s_dist, pre_neighbor_mean_dist, \
               pre_neighbor_std_dist, mean_d, std_d

    def save_model(self):
        self.model_update_queue_set.flag_queue.put(ModelUpdateQueueSet.SAVE)

    def infer_embeddings(self, data):
        self.infer_model.to(self.device)
        data = torch.tensor(data, dtype=torch.float).to(self.device)
        with torch.no_grad():
            self.infer_model.eval()
            data_embeddings = self.infer_model(data).cpu()

        return data_embeddings.numpy()

    def embed_current_data(self):
        embeddings = self.infer_embeddings(self.stream_dataset.get_total_data())
        return embeddings

    def embed_updating_collected_data(self):
        data = self.stream_dataset.get_total_data()[self.data_num_when_update:]
        return self.infer_embeddings(data)

    def update_scdr(self, newest_model, embeddings, stream_dataset, first=False):
        self.stream_dataset = stream_dataset
        self.infer_model = newest_model
        self.infer_model = self.infer_model.to(self.device)
        self.fitted_data_num = embeddings.shape[0]
        self._neighbor_changed_indices.clear()
        if first:
            self.model_trained = True

    def update_thresholds(self, cluster_indices):
        pre_neighbor_embedding_m_dist, pre_neighbor_embedding_s_dist, pre_neighbor_mean_dist, \
        pre_neighbor_std_dist, mean_dist2clusters, std_dist2clusters = self.get_pre_embedding_statistics(
            cluster_indices)

        d_thresh = mean_dist2clusters + 1 * std_dist2clusters
        e_thresh = pre_neighbor_embedding_m_dist + 0 * pre_neighbor_embedding_s_dist
        self.embedding_quality_supervisor.update_d_e_thresh(d_thresh, e_thresh)
        # print("d thresh:", d_thresh, " e thresh:", e_thresh)

        local_move_thresh = pre_neighbor_embedding_m_dist + 3 * pre_neighbor_embedding_s_dist
        bfgs_update_thresh = pre_neighbor_mean_dist + 1 * pre_neighbor_std_dist
        self.embedding_optimizer.update_local_move_thresh(local_move_thresh)
        self.embedding_optimizer.update_bfgs_update_thresh(bfgs_update_thresh)

    def ending(self):
        total_time = self.knn_cal_time + self.knn_update_time + self.model_infer_time + self.quality_record_time + \
                     self.embedding_opt_time + self.embedding_update_time
        print("kNN Cal: %.4f kNN Update: %.4f Model Initial: %.4f Model Infer: %.4f Quality Record: %.4f"
              " Embedding Opt: %.4f Embedding Update: %.4f Total: %.4f" %
              (self.knn_cal_time, self.knn_update_time, self.model_init_time, self.model_infer_time,
               self.quality_record_time, self.embedding_opt_time, self.embedding_update_time, total_time))


def query_knn(query_data, data_set, k, return_indices=False):
    dists = cdist(query_data, data_set)
    sort_indices = np.argsort(dists, axis=1)
    knn_indices = sort_indices[:, 1:k + 1]

    knn_distances = []
    for i in range(knn_indices.shape[0]):
        knn_distances.append(dists[i, knn_indices[i]])
    knn_distances = np.array(knn_distances)
    if return_indices:
        return knn_indices, knn_distances, sort_indices
    return knn_indices, knn_distances
