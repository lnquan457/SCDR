import time
from multiprocessing import Process

import numpy as np
import torch
from scipy.spatial.distance import cdist

from model.scdr.dependencies.embedding_optimizer import NNEmbedder, EmbeddingOptimizer
from model.scdr.dependencies.scdr_utils import EmbeddingQualitySupervisor
from utils.nn_utils import StreamingANNSearchAnnoy, compute_knn_graph
from utils.queue_set import ModelUpdateQueueSet, DataProcessorQueue

OPTIMIZE_NEW_DATA_EMBEDDING = False
OPTIMIZE_NEIGHBORS = True


class DataProcessor:
    def __init__(self, n_neighbors, batch_size, model_update_queue_set, window_size, device="cuda:0", serialization=False):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.model_update_queue_set = model_update_queue_set
        self.device = device
        self._window_size = window_size
        self.nn_embedder = None

        self.serialization = serialization
        self.data_num_when_update = 0

        # 用于确定何时要更新模型，1）新的流形出现；2）模型对旧流形数据的嵌入质量太低；3）长时间没有更新；
        self.model_update_intervals = 6000
        self.model_update_num_thresh = 50
        self.manifold_change_num_thresh = 100
        self.bad_embedding_num_thresh = 100

        # self._manifold_change_d_weight = 1
        # self._local_move_std_weight = 3
        self._do_model_update = True
        # 是否进行跳步优化
        self.skip_opt = True
        # bfgs优化时,使用的负例数目
        self.opt_neg_num = 50

        # 模型优化时，对于旧流形中的数据，采样的比例
        self._old_manifold_sample_rate = 1.0  # 0.5
        self._min_training_num = 100

        self.update_when_end = False

        self.knn_searcher_approx = StreamingANNSearchAnnoy()
        self._knn_update_iter = 1000

        # 进行初次训练后，初始化以下对象
        self.stream_dataset = None
        self.embedding_quality_supervisor = None
        self._opt_judge_acc = 0
        self._change_judge_acc = 0
        self.embedding_optimizer = None

        self.initial_data_buffer = None
        self.initial_label_buffer = None

        self.fitted_data_num = 0
        self.current_model_unfitted_num = 0
        self._need_replace_model = False
        self._update_after_replace_signal = False
        self._model_just_replaced = False
        self._model_is_updating = False
        self._model_update_delayed = False
        self._last_update_meta = None

        self._total_processed_data_num = 0
        self._out_since_last_send_update = 0
        self._skipped_slide_num = []
        self._cached_candidate_indices = None
        self._cached_candidate_dists = None
        self._cached_candidate_idx = None
        self._candidate_num = 0
        self._process_num = 0

        self.whole_time = 0
        self.knn_cal_time = 0
        self.knn_update_time = 0
        self.model_init_time = 0
        self.replace_model_time = 0
        self.embedding_opt_time = 0
        self.embedding_update_time = 0
        self.quality_record_time = 0
        self.update_model_time = 0
        self._slide_time = 0
        self.update_count = 0
        self.opt_count = 0

        self._record_time = True
        self.debug = True

    def init_stream_dataset(self, stream_dataset):
        self.stream_dataset = stream_dataset
        self.stream_dataset.initial_data_num = self.stream_dataset.get_n_samples()
        self._total_processed_data_num = self.stream_dataset.get_n_samples()
        self.fitted_data_num = self.stream_dataset.get_n_samples()
        self._candidate_num = (self.knn_searcher_approx._beta - 1) * self.n_neighbors
        self._cached_candidate_indices = np.ones((self.fitted_data_num, self._candidate_num), dtype=int) * -1
        self._cached_candidate_dists = np.ones((self.fitted_data_num, self._candidate_num), dtype=int) * -1
        self._cached_candidate_idx = [1] * len(self._cached_candidate_indices)
        # self._model_just_replaced = True
        self._is_new_manifold = [False] * self.fitted_data_num
        self.initial_embedding_optimizer_and_quality_supervisor()

    # scdr本身属于嵌入进程，负责数据的处理和嵌入
    def process(self, data, data_embeddings, labels=None):
        # w_sta = time.time()
        other_time = 0
        self._process_num += 1
        self._total_processed_data_num += data.shape[0]
        pre_n_samples = self.stream_dataset.get_n_samples()
        pre_embeddings = self.stream_dataset.get_total_embeddings()

        # if self._record_time:
        #     sta = time.time()
        update = False
        if self._model_just_replaced:
            update = True
            self._model_just_replaced = False

        knn_indices, knn_dists, candidate_indices, candidate_dists = \
            self.knn_searcher_approx.search_2(self.n_neighbors, pre_embeddings,
                                              self.stream_dataset.get_total_data(),
                                              data_embeddings, data, self.current_model_unfitted_num, update)

        self._cached_candidate_idx.append(0)
        self._cached_candidate_indices = \
            np.concatenate([self._cached_candidate_indices, candidate_indices[0][np.newaxis, self.n_neighbors:self.n_neighbors + self._candidate_num]], axis=0)
        self._cached_candidate_dists = \
            np.concatenate([self._cached_candidate_dists, candidate_dists[0][np.newaxis,
                                                          self.n_neighbors:self.n_neighbors + self._candidate_num]], axis=0)

        # knn_indices = knn_indices[np.newaxis, :]
        # knn_dists = knn_dists[np.newaxis, :]

        # 准确查询
        if self.debug:
            knn_indices, knn_dists = query_knn2(data, np.concatenate([self.stream_dataset.get_total_data(), data],
                                                                    axis=0), k=self.n_neighbors)
        # if self._record_time:
        #     self.knn_cal_time += time.time() - sta
        # print("knn acc:", len(np.intersect1d(knn_indices.squeeze(), acc_knn_indices.squeeze()))/self.n_neighbors)

        sta = time.time()
        self.stream_dataset.add_new_data(data, None, labels, knn_indices, knn_dists)
        other_time += time.time() - sta

        # acc_knn_indices, acc_knn_dists = compute_knn_graph(self.stream_dataset.get_total_data(), None, 10, None)
        # cur_knn_indices = self.stream_dataset.get_knn_indices()
        # before_acc = np.sum(np.ravel(acc_knn_indices) == np.ravel(cur_knn_indices)) / len(np.ravel(acc_knn_indices))
        # print("before_acc", before_acc)

        # if self._record_time:
        #     sta = time.time()

        self.stream_dataset.update_knn_graph(pre_n_samples, data, None, candidate_indices, candidate_dists,
                                             update_similarity=False, symmetric=False)
        # if self._record_time and self.embedding_update_time > 0:
        #     self.knn_update_time += time.time() - sta

        if self._process_num % self._knn_update_iter == 0:
            acc_knn_indices, acc_knn_dists = compute_knn_graph(self.stream_dataset.get_total_data(), None,
                                                               self.n_neighbors, None)
            self.stream_dataset._knn_manager.update_knn_graph(acc_knn_indices, acc_knn_dists)

        if update:
            fit_data = [self.stream_dataset.get_knn_indices(), self.stream_dataset.get_knn_dists(),
                        self.fitted_data_num]
        else:
            fit_data = None

        # neighbor_embeddings = None
        # if self._record_time:
        #     sta = time.time()
        # neighbor_embeddings = self._model_embeddings[knn_indices.squeeze()]
        # if self._record_time:
        #     self.model_infer_time += time.time() - sta

        # if self._record_time:
        #     sta = time.time()
        # p_need_optimize, manifold_change, need_replace_model, need_update_model = \
        #     self.embedding_quality_supervisor.quality_record_lof(data_embeddings, neighbor_embeddings,
        #                                                          knn_indices, knn_dists, fit_data)

        p_need_optimize, manifold_change, need_replace_model, need_update_model = \
            self.embedding_quality_supervisor.quality_record_simple(knn_indices, knn_dists, fit_data)

        # self._is_embedding_optimized = np.append(self._is_embedding_optimized, p_need_optimize)
        # print(p_need_optimize)
        # if self._record_time:
        #     self.quality_record_time += time.time() - sta

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

        # if not need_replace_model:
        #     self.stream_dataset.add_new_data(embeddings=data_embeddings)
        # else:
        #     self.stream_dataset.get_total_embeddings()[pre_n_samples:] = data_embeddings
        sta = time.time()
        self.stream_dataset.add_new_data(embeddings=data_embeddings)
        other_time += time.time() - sta

        if self._do_model_update and (need_update_model or self._model_update_delayed):
            self._send_update_signal()

        need_replace_model = need_replace_model and self._last_update_meta is not None
        if self._do_model_update and (need_replace_model or self._need_replace_model):
            if self.model_update_queue_set.training_data_queue.empty() or self._update_after_replace_signal:
                # print("replace model")
                # if self._record_time:
                #     sta = time.time()
                self._replace_model()
                need_replace_model = True
                # if self._record_time:
                #     self.replace_model_time += time.time() - sta
            else:
                self._need_replace_model = True
                need_replace_model = False

        # if not need_replace_model:
        #     pass
            # ====================================1. 只对新数据本身的嵌入进行更新=======================================
            # if OPTIMIZE_NEW_DATA_EMBEDDING:
            #     self.opt_count += 1
            #     # print("opt count", self.opt_count)
            #     if self._record_time:
            #         sta = time.time()
            #
            #     # ===========================================for batch process========================================
            #     for i in range(data_embeddings.shape[0]):
            #         if p_need_optimize[i]:
            #             data_embeddings[i] = self.embedding_optimizer.optimize_new_data_embedding(
            #                 self.stream_dataset.raw_knn_weights[pre_n_samples+i], neighbor_embeddings, pre_embeddings)
            #     # ===========================================for batch process========================================
            #
            #     if self._record_time:
            #         self.embedding_opt_time += time.time() - sta
            # =====================================================================================================

        # if not need_replace_model:
        #     self.stream_dataset.add_new_data(embeddings=data_embeddings)
        # else:
        #     self.stream_dataset.get_total_embeddings()[pre_n_samples:] = data_embeddings

        if not need_replace_model:
            # if self._record_time:
            #     sta = time.time()
            neighbor_changed_indices, replaced_raw_weights, replaced_indices, anchor_positions = \
                self.stream_dataset.get_pre_neighbor_changed_info()
            if len(neighbor_changed_indices) > 0:
                # self._neighbor_changed_indices = self._neighbor_changed_indices.union(neighbor_changed_indices)
                optimized_embeddings = self.embedding_optimizer.update_old_data_embedding(
                    data.shape[0], self.stream_dataset.get_total_embeddings(), neighbor_changed_indices,
                    self.stream_dataset.get_knn_indices(), self.stream_dataset.get_knn_dists(),
                    self.stream_dataset.raw_knn_weights[neighbor_changed_indices], anchor_positions,
                    replaced_indices, replaced_raw_weights)
                self.stream_dataset.update_embeddings(optimized_embeddings)
            # if self._record_time:
            #     self.embedding_update_time += time.time() - sta

        ret = self.stream_dataset.get_total_embeddings()
        # self.whole_time += time.time() - w_sta
        # print("whole_time", self.whole_time)
        return ret, other_time

    def _replace_model(self):
        # Todo: 这里进行替换时，不一定是最新的模型

        newest_model, embeddings, total_data_idx = self._last_update_meta
        self.nn_embedder.update_model(newest_model)
        # 最简单的方式就是直接使用模型嵌入，但是如果这些数据来自新的流形，就可能会破坏稳定性并且降低质量。
        new_data_embeddings = self.embed_updating_collected_data(total_data_idx)
        total_embeddings = np.concatenate([embeddings, new_data_embeddings], axis=0)

        self.stream_dataset.update_embeddings(total_embeddings)
        self.stream_dataset.update_unfitted_data_num(new_data_embeddings.shape[0])
        # self.update_thresholds()
        self._model_embeddings = total_embeddings
        self._model_just_replaced = True
        self.current_model_unfitted_num = new_data_embeddings.shape[0]
        self._last_update_meta = None
        self._need_replace_model = False
        self._update_after_replace_signal = False

    def _send_update_signal(self):

        # 当前还有模型更新任务正在进行时，可以暂缓模型更新。
        if self._model_is_updating:
            self._model_update_delayed = True
            return

        # sta = time.time()
        # acc_knn_indices, _ = compute_knn_graph(self.stream_dataset.get_total_data(), None, self.n_neighbors, None)
        # acc = np.ravel(self.stream_dataset.get_knn_indices()) == np.ravel(acc_knn_indices)
        # print("kNN acc:", np.sum(acc) / len(acc))
        # self.whole_time -= time.time() - sta

        pre_fitted_num = self.fitted_data_num
        self.fitted_data_num = self.stream_dataset.get_n_samples()

        while not self.model_update_queue_set.training_data_queue.empty():
            self.model_update_queue_set.training_data_queue.get()
        print("send fitted data num:", pre_fitted_num)
        self.model_update_queue_set.training_data_queue.put(
            [self.stream_dataset, None, pre_fitted_num,
             self.stream_dataset.get_n_samples() - pre_fitted_num,
             self._total_processed_data_num, self._out_since_last_send_update])
        self._model_is_updating = True
        self.model_update_queue_set.MODEL_UPDATING.value = 1
        self.model_update_queue_set.flag_queue.put(ModelUpdateQueueSet.UPDATE)
        self.update_count += 1
        self._model_update_delayed = False
        self._out_since_last_send_update = 0

        # TODO: 使得模型更新和数据处理变成串行的
        if self.serialization:
            while self.model_update_queue_set.MODEL_UPDATING.value == 1:
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
            self.update_scdr(infer_model, embeddings, stream_dataset, self._total_processed_data_num)
        return embeddings

    def initial_embedding_optimizer_and_quality_supervisor(self):
        pre_neighbor_embedding_m_dist, pre_neighbor_embedding_s_dist = \
            self.stream_dataset.get_embedding_neighbor_mean_std_dist()

        # =====================================初始化embedding_quality_supervisor===================================
        e_thresh = pre_neighbor_embedding_m_dist + 3 * pre_neighbor_embedding_s_dist
        self.embedding_quality_supervisor = EmbeddingQualitySupervisor(self.model_update_intervals,
                                                                       self.manifold_change_num_thresh,
                                                                       self.bad_embedding_num_thresh,
                                                                       self.model_update_num_thresh,
                                                                       e_thresh)
        self.embedding_quality_supervisor._lof.my_fit(self.stream_dataset.get_knn_indices(),
                                                      self.stream_dataset.get_knn_dists(),
                                                      self.stream_dataset.get_n_samples())
        self.embedding_quality_supervisor.update_model_update_time(time.time())
        # ==========================================================================================================

        # =====================================初始化embedding_optimizer==============================================
        self.embedding_optimizer = EmbeddingOptimizer(neg_num=self.opt_neg_num, skip_opt=self.skip_opt)
        # ===========================================================================================================

    def infer_embeddings(self, data):
        return self.nn_embedder.embed(data)

    def embed_updating_collected_data(self, data_num_when_update):
        data = self.stream_dataset.get_total_data()[-(self._total_processed_data_num - data_num_when_update):]
        return self.infer_embeddings(data)

    def update_scdr(self, newest_model, embeddings, stream_dataset, total_data_idx):
        sta = time.time()
        pre_embeddings = self.stream_dataset.get_total_embeddings()
        # self.stream_dataset = stream_dataset
        self.stream_dataset.update_previous_info(embeddings.shape[0], stream_dataset,
                                                 self._out_since_last_send_update,
                                                 self._skipped_slide_num[0] if len(self._skipped_slide_num) > 0 else 0)
        self._last_update_meta = [newest_model, embeddings, total_data_idx]
        self.stream_dataset.update_embeddings(pre_embeddings)
        self._skipped_slide_num = []
        self._model_is_updating = False
        if self._need_replace_model:
            self._update_after_replace_signal = True
        self.update_model_time += time.time() - sta

        return pre_embeddings

    def update_thresholds(self):
        sta = time.time()
        # 只在替换模型时，才更新这些阈值，导致滞后性比较严重。
        pre_neighbor_embedding_m_dist, pre_neighbor_embedding_s_dist = \
            self.stream_dataset.get_embedding_neighbor_mean_std_dist()

        if self.embedding_quality_supervisor is not None:
            e_thresh = pre_neighbor_embedding_m_dist + 3 * pre_neighbor_embedding_s_dist
            self.embedding_quality_supervisor.update_threshes(e_thresh)
            # print("d thresh:", d_scale_low, d_scale_high)

        self.update_model_time += time.time() - sta

    def ending(self):
        total_time = self.knn_cal_time + self.knn_update_time + self.quality_record_time + \
                     self.embedding_opt_time + self.embedding_update_time + self.update_model_time + self.replace_model_time
        output = "kNN Cal: %.4f kNN Update: %.4f Quality Record: %.4f " \
                 "Embedding Opt: %.4f Embedding Update: %.4f Update Model: %.4f Replace: %.4f Slide: %.4f Total: %.4f " \
                 % (self.knn_cal_time, self.knn_update_time,
                    self.quality_record_time, self.embedding_opt_time, self.embedding_update_time,
                    self.update_model_time, self.replace_model_time, self._slide_time, total_time)
        print(output)
        return output


class DataProcessorProcess(DataProcessor, Process):
    def __init__(self, embedding_data_queue, n_neighbors, batch_size,
                 model_update_queue_set, window_size, device="cuda:0", serialization=False):
        self.name = "data update process"
        DataProcessor.__init__(self, n_neighbors, batch_size, model_update_queue_set, window_size, device, serialization)
        Process.__init__(self, name=self.name)
        self._embedding_data_queue: DataProcessorQueue = embedding_data_queue
        self._newest_model = None
        self.sub_time = 0
        self.dataset_slide_time = 0
        self.last_time = 0

    def run(self) -> None:
        while True:

            if not self.model_update_queue_set.embedding_queue.empty():
                embeddings, infer_model, stream_dataset, total_data_idx = \
                    self.model_update_queue_set.embedding_queue.get()
                self.update_scdr(infer_model, embeddings, stream_dataset, total_data_idx)

            data, data_embedding, label, is_end = self._embedding_data_queue.get()

            if is_end:
                break

            self._embedding_data_queue.processing()
            self._newest_model = None
            # TODO: 滑动窗口
            sta = time.time()
            out_num = self.slide_window()
            self._slide_time += time.time() - sta

            # sta = time.time()
            total_embeddings, other_time = super().process(data, data_embedding, label)
            self._embedding_data_queue.put_res([total_embeddings, self._newest_model, other_time, out_num])
            self._embedding_data_queue.processed()

        self.ending()

    def _replace_model(self):
        # Todo: 这里进行替换时，不一定是最新的模型
        print("replace model!")
        newest_model, embeddings, total_data_idx = self._last_update_meta
        self._newest_model = newest_model

        data = self.stream_dataset.get_total_data()[-(self._total_processed_data_num - total_data_idx):]
        data = torch.tensor(data, dtype=torch.float)
        with torch.no_grad():
            tmp_model = newest_model
            new_data_embeddings = tmp_model(data).numpy()

        total_embeddings = np.concatenate([embeddings, new_data_embeddings], axis=0)[
                           -self.stream_dataset.get_n_samples():]
        self.stream_dataset.update_embeddings(total_embeddings)
        self.stream_dataset.update_unfitted_data_num(new_data_embeddings.shape[0])
        # self.update_thresholds()
        self._model_just_replaced = True
        self.current_model_unfitted_num = new_data_embeddings.shape[0]
        self._last_update_meta = None
        self._need_replace_model = False
        self._update_after_replace_signal = False

    def slide_window(self):
        out_num = max(0, self.stream_dataset.get_n_samples() - self._window_size)
        if out_num <= 0:
            return out_num

        # sta = time.time()
        self._cached_candidate_indices = self._cached_candidate_indices[out_num:]
        self._cached_candidate_indices -= out_num
        self._cached_candidate_dists = self._cached_candidate_dists[out_num:]
        self._cached_candidate_idx = self._cached_candidate_idx[out_num:]

        self.stream_dataset.slide_window(out_num, self._cached_candidate_indices, self._cached_candidate_dists,
                                         self._cached_candidate_idx)

        # sta = time.time()
        self.embedding_quality_supervisor.slide_window(out_num)
        self._is_new_manifold = self._is_new_manifold[out_num:]
        self.fitted_data_num = max(0, self.fitted_data_num - out_num)
        self._out_since_last_send_update += out_num

        if self._model_is_updating and len(self._skipped_slide_num) == 0:
            self._skipped_slide_num.append(out_num)

        return out_num


def query_knn2(query_data, data_set, k, return_indices=False):
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
