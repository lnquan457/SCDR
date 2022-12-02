import os.path
import time

import numpy as np
import torch
from scipy.spatial.distance import cdist

from dataset.warppers import StreamingDatasetWrapper, DataRepo
from model.scdr.dependencies.embedding_optimizer import EmbeddingOptimizer
from utils.logger import InfoLogger
from utils.nn_utils import StreamingANNSearchKD, compute_knn_graph, StreamingANNSearchAnnoy
from utils.queue_set import ModelUpdateQueueSet, DataProcessorQueueSet
from model.scdr.dependencies.scdr_utils import KeyPointsGenerator, DistributionChangeDetector, ClusterRepDataSampler, \
    EmbeddingQualitySupervisor

OPTIMIZE_NEW_DATA_EMBEDDING = True
OPTIMIZE_NEIGHBORS = True


def _sample_training_data(fitted_data_num, n_samples, _is_new_manifold, min_num, sample_rate):
    new_data_indices = np.arange(fitted_data_num, n_samples)
    if sample_rate >= 1 or min_num >= len(new_data_indices) * sample_rate:
        return new_data_indices
    new_manifold_data_indices = new_data_indices[_is_new_manifold]
    old_manifold_data_indices = np.setdiff1d(new_data_indices, new_manifold_data_indices)
    np.random.shuffle(old_manifold_data_indices)
    sample_num = max(min_num - len(new_manifold_data_indices),
                     int(len(old_manifold_data_indices) * sample_rate))
    sample_indices = np.concatenate([new_manifold_data_indices,
                                     old_manifold_data_indices[:min(sample_num, len(old_manifold_data_indices))]])
    return sample_indices


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
        self.model_update_intervals = 6000
        # TODO: 这个过程中，保持的knn可能是不对的，在之后也无法得到修正。所以这个值不能设置的太小了。
        self.model_update_num_thresh = 100
        self.manifold_change_num_thresh = 200
        self.bad_embedding_num_thresh = 400

        self._manifold_change_d_weight = 1
        self._local_move_std_weight = 3
        # 是否进行跳步优化
        self.skip_opt = True
        # bfgs优化时,使用的负例数目
        self.opt_neg_num = 50

        # 模型优化时，对于旧流形中的数据，采样的比例
        self._old_manifold_sample_rate = 0.5
        self._min_training_num = 100
        # 模型优化时，每个batch中的旧数据采样率
        self.rep_data_sample_rate = 0.07
        self.rep_data_minimum_num = 50
        # 是否要采样所有旧数据
        self.cover_all = True
        self.update_when_end = False

        # self.knn_searcher_approx = StreamingANNSearchKD()
        self.knn_searcher_approx = StreamingANNSearchAnnoy()

        self.cluster_rep_data_sampler = ClusterRepDataSampler(self.rep_data_sample_rate, self.rep_data_minimum_num,
                                                              self.cover_all)
        self.pre_cluster_centers = None
        # 进行初次训练后，初始化以下对象
        self.stream_dataset = StreamingDatasetWrapper(batch_size, n_neighbors, self.device)
        self.embedding_quality_supervisor = None
        self._opt_judge_acc = 0
        self._change_judge_acc = 0
        self.embedding_optimizer = None

        self.initial_data_buffer = None
        self.initial_label_buffer = None

        self._model_embeddings = None
        self.model_trained = False
        self.fitting_data_num = 0
        self.fitted_data_num = 0
        self.current_model_fitted_num = 0
        self._need_replace_model = False
        self._update_after_replace_signal = False
        self.model_just_replaced = False
        # self._neighbor_changed_indices = set()
        self._last_update_meta = None
        # 新数据是否来自新的流形，以及其嵌入是否经过优化
        self._is_new_manifold = []
        self._is_embedding_optimized = []

        self.whole_time = 0
        self.knn_cal_time = 0
        self.knn_update_time = 0
        self.model_init_time = 0
        self.model_infer_time = 0
        self.replace_model_time = 0
        self.embedding_opt_time = 0
        self.embedding_update_time = 0
        self.quality_record_time = 0
        self.update_model_time = 0
        self.update_count = 0
        self.opt_count = 0

        self._record_time = True
        self.debug = True

    # scdr本身属于嵌入进程，负责数据的处理和嵌入
    def fit_new_data(self, data, labels=None):
        w_sta = time.time()
        # new_data_num = data.shape[0]
        pre_n_samples = self.stream_dataset.get_n_samples() - 1
        # if self.model_trained:
        #     print("pre embedding samples", self.stream_dataset.get_total_embeddings().shape[0], pre_n_samples)

        if not self.model_trained:
            if not self._caching_initial_data(data, labels):
                return None
            sta = time.time()
            self.stream_dataset.add_new_data(data=self.initial_data_buffer, labels=self.initial_label_buffer)
            cluster_indices = self._initial_project_model()
            self._initial_embedding_optimizer_and_quality_supervisor(cluster_indices)
            if self._record_time:
                self.model_init_time += time.time() - sta
            self.whole_time -= time.time() - sta
        else:
            pre_embeddings = self.stream_dataset.get_total_embeddings()
            # InfoLogger.info("embedding num: {} pre n samples: {}".format(pre_embeddings.shape[0], pre_n_samples))
            # 使用之前的投影函数对最新的数据进行投影
            if self._record_time:
                sta = time.time()
            data_embeddings = self.infer_embeddings(data)
            self._model_embeddings = np.concatenate([self._model_embeddings, data_embeddings], axis=0)
            if self._record_time:
                self.model_infer_time += time.time() - sta

            if self._record_time:
                sta = time.time()
            update = False
            if self.model_just_replaced:
                update = True
                self.model_just_replaced = False
            # pre_optimized = self._is_embedding_optimized[-1] if len(self._is_embedding_optimized) > 0 else None
            # knn_indices, knn_dists = \
            #     self.knn_searcher_approx.search(self.n_neighbors, pre_embeddings, self.stream_dataset.get_total_data(),
            #                                     data_embeddings, data, pre_optimized, update)
            knn_indices, knn_dists, candidate_indices, candidate_dists = \
                self.knn_searcher_approx.search_2(self.n_neighbors, pre_embeddings,
                                                  self.stream_dataset.get_total_data()[:-1],
                                                  self.current_model_fitted_num, data_embeddings, data, update)
            knn_indices = knn_indices[np.newaxis, :]
            knn_dists = knn_dists[np.newaxis, :]

            # 准确查询
            # knn_indices, knn_dists = query_knn(data, np.concatenate([self.stream_dataset.get_total_data(), data],
            #                                                         axis=0), k=self.n_neighbors)
            if self._record_time:
                self.knn_cal_time += time.time() - sta
            # print("knn acc:", len(np.intersect1d(knn_indices.squeeze(), acc_knn_indices.squeeze()))/self.n_neighbors)
            # Todo: 这里非常耗时
            self.stream_dataset.add_new_data(None, None, labels, knn_indices, knn_dists)

            if self._record_time:
                sta = time.time()
            self.stream_dataset.update_knn_graph(pre_n_samples, data, None, candidate_indices, candidate_dists,
                                                 update_similarity=False, symmetric=False)
            if self._record_time and self.embedding_opt_time > 0:
                self.knn_update_time += time.time() - sta

            if self._record_time:
                sta = time.time()
            neighbor_embeddings = self._model_embeddings[knn_indices.squeeze()]
            if self._record_time:
                self.model_infer_time += time.time() - sta

            if self._record_time:
                sta = time.time()
            # p_need_optimize, manifold_change, need_update_model = \
            #     self.embedding_quality_supervisor.quality_record(data, data_embeddings, self.pre_cluster_centers,
            #                                                      neighbor_embeddings)
            # p_need_optimize, manifold_change, need_update_model = \
            #     self.embedding_quality_supervisor.quality_record_2(data, data_embeddings, knn_dists,
            #                                                        neighbor_embeddings)
            fit_data = self.stream_dataset.get_total_data()[:self.current_model_fitted_num] if update else None
            p_need_optimize, manifold_change, need_replace_model, need_update_model = \
                self.embedding_quality_supervisor.quality_record_lof(data, data_embeddings, neighbor_embeddings,
                                                                     knn_indices, knn_dists, fit_data)

            self._is_new_manifold.append(manifold_change)
            self._is_embedding_optimized.append(p_need_optimize)
            # print(p_need_optimize)
            if self._record_time:
                self.quality_record_time += time.time() - sta

            # unique_labels, counts = np.unique(self.stream_dataset.get_total_label()[:self.fitted_data_num],
            #                                   return_counts=True)
            # tt_indices = np.where(counts > 50)[0]
            # need_optimize = labels not in (unique_labels[tt_indices])
            # if need_optimize == p_need_optimize:
            #     self._opt_judge_acc += 1
            # if need_optimize == manifold_change:
            #     self._change_judge_acc += 1
            # print(labels, unique_labels[tt_indices])
            # print(p_need_optimize, need_optimize, " supervise acc:",
            #       self._opt_judge_acc / (self.stream_dataset.get_n_samples() - 861))
            # print(manifold_change, need_optimize, " manifold change acc:",
            #       self._change_judge_acc / (self.stream_dataset.get_n_samples() - 861))

            if need_update_model:
                self._send_update_signal()

            need_replace_model = need_replace_model and self._last_update_meta is not None
            if need_replace_model or self._need_replace_model:
                if self.model_update_queue_set.training_data_queue.empty() or self._update_after_replace_signal:
                    print("replace model")
                    if self._record_time:
                        sta = time.time()
                    # Todo: 这里进行替换时，不一定是最新的模型
                    newest_model, embeddings = self._last_update_meta
                    self.infer_model = newest_model
                    self.infer_model = self.infer_model.to(self.device)
                    # 最简单的方式就是直接使用模型嵌入，但是如果这些数据来自新的流形，就可能会破坏稳定性并且降低质量。
                    new_data_embeddings = self.embed_updating_collected_data(embeddings.shape[0])
                    total_embeddings = np.concatenate([embeddings, new_data_embeddings], axis=0)
                    self.stream_dataset.update_embeddings(total_embeddings)
                    self.stream_dataset.update_fitted_data_num(embeddings.shape[0])
                    self.update_thresholds(embeddings.shape[0], None)
                    # self.stream_dataset.update_embeddings(total_embeddings)
                    self._model_embeddings = total_embeddings
                    self.model_just_replaced = True
                    self.current_model_fitted_num = embeddings.shape[0]
                    self._last_update_meta = None
                    self._need_replace_model = False
                    self._update_after_replace_signal = False
                    need_replace_model = True
                    if self._record_time:
                        self.replace_model_time += time.time() - sta
                else:
                    self._need_replace_model = True
                    need_replace_model = False

            if not need_replace_model and p_need_optimize:
                # ====================================1. 只对新数据本身的嵌入进行更新=======================================
                if OPTIMIZE_NEW_DATA_EMBEDDING:
                    self.opt_count += 1
                    # print("opt count", self.opt_count)
                    if self._record_time:
                        sta = time.time()
                    data_embeddings = self.embedding_optimizer.optimize_new_data_embedding(
                        self.stream_dataset.raw_knn_weights[self.stream_dataset.get_n_samples() - 1],
                        neighbor_embeddings, pre_embeddings)
                    if self._record_time:
                        self.embedding_opt_time += time.time() - sta
                # =====================================================================================================

            if not need_replace_model:
                self.stream_dataset.add_new_data(embeddings=data_embeddings)
            else:
                self.stream_dataset.get_total_embeddings()[-1] = data_embeddings

            if OPTIMIZE_NEIGHBORS and not need_replace_model:
                if self._record_time:
                    sta = time.time()
                neighbor_changed_indices, replaced_raw_weights, replaced_indices, anchor_positions = \
                    self.stream_dataset.get_pre_neighbor_changed_info()
                if len(neighbor_changed_indices) > 0:
                    # self._neighbor_changed_indices = self._neighbor_changed_indices.union(neighbor_changed_indices)
                    optimized_embeddings = self.embedding_optimizer.update_old_data_embedding(
                        data_embeddings, self.stream_dataset.get_total_embeddings(), neighbor_changed_indices,
                        self.stream_dataset.get_knn_indices(), self.stream_dataset.get_knn_dists(),
                        self.stream_dataset.raw_knn_weights[neighbor_changed_indices], anchor_positions,
                        replaced_indices, replaced_raw_weights)
                    self.stream_dataset.update_embeddings(optimized_embeddings)
                if self._record_time:
                    self.embedding_update_time += time.time() - sta

        ret = self.stream_dataset.get_total_embeddings()
        self.whole_time += time.time() - w_sta
        # print("whole_time", self.whole_time)
        return ret

    def _send_update_signal(self):

        # sta = time.time()
        # acc_knn_indices, _ = compute_knn_graph(self.stream_dataset.get_total_data(), None, self.n_neighbors, None)
        # acc = np.ravel(self.stream_dataset.get_knn_indices()) == np.ravel(acc_knn_indices)
        # print("kNN acc:", np.sum(acc) / len(acc))
        # self.whole_time -= time.time() - sta

        # if self._record_time:
        #     sta = time.time()
        # self.stream_dataset.update_cached_neighbor_similarities()
        # if self.update_count > 0:
        #     self.knn_update_time += time.time() - sta
        # 如果还有模型更新任务没有完成，此处应该阻塞等待
        sta = time.time()

        # sample_indices = _sample_training_data(self.fitted_data_num, self.stream_dataset.get_n_samples(),
        #                                        self._is_new_manifold, self._min_training_num, self._old_manifold_sample_rate)

        pre_fitted_num = self.fitted_data_num
        self.fitted_data_num = self.stream_dataset.get_n_samples()
        # print("send signal . curren data num", self.fitted_data_num)

        # while self.model_update_queue_set.MODEL_UPDATING.value == 1:
        #     pass
        # self.model_update_queue_set.MODEL_UPDATING.value = 1

        while not self.model_update_queue_set.training_data_queue.empty():
            self.model_update_queue_set.training_data_queue.get()
        # self.model_update_queue_set.training_data_queue.clear()

        self.model_update_queue_set.training_data_queue.put(
            [self.stream_dataset, self.cluster_rep_data_sampler, pre_fitted_num,
             self.stream_dataset.get_n_samples() - pre_fitted_num, self._is_new_manifold])
        self.model_update_queue_set.flag_queue.put(ModelUpdateQueueSet.UPDATE)
        self.update_count += 1

        # if self._record_time:
        #     self.update_model_time += time.time() - sta

        # TODO: 使得模型更新和数据处理变成串行的
        # while self.model_update_queue_set.MODEL_UPDATING.value:
        #     pass
        # self.model_update_queue_set.WAITING_UPDATED_DATA.value = 1

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
        # self.data_num_when_update = self.stream_dataset.get_total_data().shape[0]
        self.model_update_queue_set.training_data_queue.put(
            [self.stream_dataset, self.cluster_rep_data_sampler, self.ckpt_path])
        self.model_update_queue_set.flag_queue.put(ModelUpdateQueueSet.UPDATE)
        self.model_update_queue_set.INITIALIZING.value = True
        # 这里传回来的stream_dataset中，包含了最新的对称kNN信息
        embeddings, model, stream_dataset, cluster_indices = self.model_update_queue_set.embedding_queue.get()
        self.stream_dataset = stream_dataset
        self.stream_dataset.update_fitted_data_num(embeddings.shape[0])
        self.infer_model = model
        self.infer_model = self.infer_model.to(self.device)
        self.fitted_data_num = embeddings.shape[0]
        self.current_model_fitted_num = self.fitted_data_num
        self._model_embeddings = embeddings
        self.model_trained = True

        self.stream_dataset.add_new_data(embeddings=embeddings)
        self.model_update_queue_set.INITIALIZING.value = False
        self.model_just_replaced = True
        return cluster_indices

    def _initial_embedding_optimizer_and_quality_supervisor(self, cluster_indices):
        pre_neighbor_embedding_m_dist, pre_neighbor_embedding_s_dist, pre_neighbor_mean_dist, \
        pre_neighbor_std_dist, mean_dist2clusters, std_dist2clusters = \
            self.get_pre_embedding_statistics(self.fitted_data_num, cluster_indices)

        # =====================================初始化embedding_quality_supervisor===================================

        d_scale_low = pre_neighbor_mean_dist - self._manifold_change_d_weight * pre_neighbor_std_dist
        d_scale_high = pre_neighbor_mean_dist + self._manifold_change_d_weight * pre_neighbor_std_dist
        # d_thresh = mean_dist2clusters + 1 * std_dist2clusters
        e_thresh = pre_neighbor_embedding_m_dist + 3 * pre_neighbor_embedding_s_dist
        print("d thresh:", d_scale_low, d_scale_high)
        self.embedding_quality_supervisor = EmbeddingQualitySupervisor(self.model_update_intervals,
                                                                       self.manifold_change_num_thresh,
                                                                       self.bad_embedding_num_thresh,
                                                                       self.model_update_num_thresh,
                                                                       [d_scale_low, d_scale_high], e_thresh)
        self.embedding_quality_supervisor.update_model_update_time(time.time())
        # ==========================================================================================================

        # =====================================初始化embedding_optimizer==============================================
        local_move_thresh = pre_neighbor_embedding_m_dist + self._local_move_std_weight * pre_neighbor_embedding_s_dist
        bfgs_update_thresh = pre_neighbor_mean_dist + 1 * pre_neighbor_std_dist
        self.embedding_optimizer = EmbeddingOptimizer(local_move_thresh, bfgs_update_thresh,
                                                      neg_num=self.opt_neg_num, skip_opt=self.skip_opt)
        # ===========================================================================================================

    def get_pre_embedding_statistics(self, fitted_num, cluster_indices=None):
        pre_neighbor_mean_dist, pre_neighbor_std_dist = self.stream_dataset.get_data_neighbor_mean_std_dist()
        pre_neighbor_embedding_m_dist, pre_neighbor_embedding_s_dist = \
            self.stream_dataset.get_embedding_neighbor_mean_std_dist()
        # print(pre_neighbor_mean_dist, pre_neighbor_std_dist)
        # print(pre_neighbor_embedding_m_dist, pre_neighbor_embedding_s_dist)
        if cluster_indices is None:
            cluster_indices = \
            self.cluster_rep_data_sampler.sample(self.stream_dataset.get_total_embeddings()[:fitted_num],
                                                 pre_neighbor_embedding_m_dist, self.n_neighbors,
                                                 self.stream_dataset.get_total_label()[:fitted_num])[-1]
        self.pre_cluster_centers, mean_d, std_d = \
            self.cluster_rep_data_sampler.dist_to_nearest_cluster_centroids(
                self.stream_dataset.get_total_data()[:fitted_num],
                cluster_indices)
        return pre_neighbor_embedding_m_dist, pre_neighbor_embedding_s_dist, pre_neighbor_mean_dist, \
               pre_neighbor_std_dist, mean_d, std_d

    def save_model(self):
        self.model_update_queue_set.flag_queue.put(ModelUpdateQueueSet.SAVE)

    def infer_embeddings(self, data):
        # self.infer_model.to(self.device)
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float, device=self.device)
        with torch.no_grad():
            # self.infer_model.eval()
            data_embeddings = self.infer_model(data).cpu()

        return data_embeddings.numpy()

    def embed_current_data(self):
        embeddings = self.infer_embeddings(self.stream_dataset.get_total_data())
        return embeddings

    def embed_updating_collected_data(self, data_num_when_update):
        data = self.stream_dataset.get_total_data()[data_num_when_update:]
        return self.infer_embeddings(data)

    def update_scdr(self, newest_model, embeddings, stream_dataset, cluster_indices=None):
        sta = time.time()
        pre_embeddings = self.stream_dataset.get_total_embeddings()
        # self.stream_dataset = stream_dataset
        self.stream_dataset.update_previous_info(embeddings.shape[0], stream_dataset)
        replace_model = True
        self._last_update_meta = [newest_model, embeddings]
        self.stream_dataset.update_embeddings(pre_embeddings)
        # self.fitted_data_num = embeddings.shape[0]
        if self._need_replace_model:
            self._update_after_replace_signal = True
        self.update_model_time += time.time() - sta

        return self.stream_dataset.get_total_embeddings(), replace_model

    def update_thresholds(self, fitted_num, cluster_indices):
        sta = time.time()
        # 只在替换模型时，才更新这些阈值，导致滞后性比较严重。
        pre_neighbor_embedding_m_dist, pre_neighbor_embedding_s_dist, pre_neighbor_mean_dist, \
        pre_neighbor_std_dist, mean_dist2clusters, std_dist2clusters = self.get_pre_embedding_statistics(fitted_num,
                                                                                                         cluster_indices)
        if self.embedding_quality_supervisor is not None:
            d_scale_low = pre_neighbor_mean_dist - self._manifold_change_d_weight * pre_neighbor_std_dist
            d_scale_high = pre_neighbor_mean_dist + self._manifold_change_d_weight * pre_neighbor_std_dist
            # d_thresh = mean_dist2clusters + 1 * std_dist2clusters
            e_thresh = pre_neighbor_embedding_m_dist + 3 * pre_neighbor_embedding_s_dist
            self.embedding_quality_supervisor.update_threshes(e_thresh, d_scale_low, d_scale_high)
            # print("d thresh:", d_scale_low, d_scale_high)

        # Todo：这里的阈值确定还要再想一想。
        # bfgs优化的结果可能比较发散，随着时间增加，阈值没有更新，实施local move的点会越来越少。
        local_move_thresh = pre_neighbor_embedding_m_dist + self._local_move_std_weight * pre_neighbor_embedding_s_dist
        # 这里用第k邻居距离比较合适
        bfgs_update_thresh = pre_neighbor_mean_dist + 1 * pre_neighbor_std_dist
        self.embedding_optimizer.update_local_move_thresh(local_move_thresh)
        self.embedding_optimizer.update_bfgs_update_thresh(bfgs_update_thresh)
        self.update_model_time += time.time() - sta

    def ending(self):
        total_time = self.knn_cal_time + self.knn_update_time + self.model_infer_time + self.quality_record_time + \
                     self.embedding_opt_time + self.embedding_update_time + self.update_model_time + self.replace_model_time
        output = "kNN Cal: %.4f kNN Update: %.4f Model Infer: %.4f Quality Record: %.4f " \
                 "Embedding Opt: %.4f Embedding Update: %.4f Update Model: %.4f Replace: %.4f Total: %.4f" \
                 % (self.knn_cal_time, self.knn_update_time, self.model_infer_time,
                    self.quality_record_time, self.embedding_opt_time, self.embedding_update_time,
                    self.update_model_time, self.replace_model_time, total_time)
        print(output)
        return output


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
