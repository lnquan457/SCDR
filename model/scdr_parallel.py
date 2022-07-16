import os.path
import random
import time

import numpy as np
import torch
from multiprocessing.managers import BaseManager
from dataset.warppers import StreamingDatasetWrapper
from utils.nn_utils import StreamingKNNSearcher, ANNOY, KD_TREE, StreamingKNNSearchApprox2, compute_knn_graph
from utils.queue_set import ModelUpdateQueueSet
from utils.scdr_utils import KeyPointsGenerater, DataSampler, DistributionChangeDetector, StreamingDataRepo


class SCDRParallel:
    def __init__(self, queue_set, shift_buffer_size, n_neighbors, batch_size, initial_train_num, initial_train_epoch,
                 finetune_epoch, ckpt_path=None, device="cuda:0"):
        self.training_queue_set = queue_set
        self.device = device
        self.batch_size = batch_size
        self.n_neighbors = n_neighbors
        self.infer_model = None
        self.ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
        self.initial_train_num = initial_train_num
        self.initial_train_epoch = initial_train_epoch
        self.finetune_epoch = finetune_epoch
        self.shift_buffer_size = shift_buffer_size

        self.initial_data_buffer = None
        self.initial_label_buffer = None
        self.pre_embeddings = None
        self.dataset = StreamingDataRepo(self.n_neighbors)

        self.cached_shift_indices = None
        self.data_num_list = [0]

        self.time_step = 0
        self.model_trained = False
        self.fitted_data_num = 0

        self.key_data = None
        self.key_data_rate = 0.5
        # self.key_data_rate = 1.0
        self.using_key_data_in_lof = True

        # 没有经过模型拟合的数据
        self.unfitted_data = None

        self.dis_change_detector = DistributionChangeDetector(lof_based=True)
        self.knn_searcher = StreamingKNNSearcher(method=ANNOY)
        self.knn_searcher_approx = StreamingKNNSearchApprox2()

        # self.total_data = None
        # self.total_label = None

        self.shift_detect_time = 0
        self.knn_update_time = 0
        self.knn_cal_time = 0
        self.model_initial_time = 0
        self.model_repro_time = 0
        self.model_update_time = 0
        self.knn_approx_time = 0
        self.debug = True

    def fit_new_data(self, data, labels=None):
        new_data_num = data.shape[0]
        if self.using_key_data_in_lof:
            self._generate_key_points(data)

        if not self.model_trained:
            if not self._caching_initial_data(data, labels):
                return None

            self.dataset.add_new_data(self.initial_data_buffer, self.initial_label_buffer)

            self._initial_project(self.initial_data_buffer, self.initial_label_buffer)

            sta = time.time()
            self.knn_searcher.search(self.initial_data_buffer, self.n_neighbors, query=False)
            self.knn_cal_time += time.time() - sta
        else:
            fit_data = self.key_data if self.using_key_data_in_lof else self.dataset.total_data
            shifted_indices = self.dis_change_detector.detect_distribution_shift(fit_data, data)
            self.cached_shift_indices = shifted_indices if self.cached_shift_indices is None else \
                np.concatenate([self.cached_shift_indices, shifted_indices])

            self.data_num_list.append(new_data_num + self.data_num_list[-1])

            # 使用之前的投影函数对最新的数据进行投影
            sta = time.time()
            data_embeddings = self.infer_embeddings(data).numpy().astype(float)
            data_embeddings = np.reshape(data_embeddings, (new_data_num, self.pre_embeddings.shape[1]))
            self.model_repro_time += time.time() - sta
            cur_total_embeddings = np.concatenate([self.pre_embeddings, data_embeddings], axis=0, dtype=float)

            sta = time.time()
            nn_indices, nn_dists = self.knn_searcher_approx. \
                search(self.n_neighbors, self.pre_embeddings[:self.fitted_data_num],
                       self.dataset.total_data[:self.fitted_data_num], data_embeddings, data, self.unfitted_data)
            self.knn_approx_time += time.time() - sta

            self.dataset.add_new_data(data, labels)
            self.training_queue_set.raw_data_queue.put([data, nn_indices, nn_dists, labels])

            if len(self.cached_shift_indices) <= self.shift_buffer_size:
                # TODO：如何计算这些最新数据的投影结果，以保证其在当前时间点是可信的，应该被当作异常点/离群点进行处理
                # TODO: 这边即使数据的分布没有发生变化，但是在投影质量下降到一定程度后，也要进行模型更新
                self.pre_embeddings = cur_total_embeddings
                self.unfitted_data = data if self.unfitted_data is None else np.concatenate([self.unfitted_data, data],
                                                                                            axis=0)
            else:
                self.pre_embeddings = self._adapt_distribution_change(cur_total_embeddings)
        return self.pre_embeddings

    def _adapt_distribution_change(self, *args):
        cur_total_embeddings = args[0]
        cur_data_num = cur_total_embeddings.shape[0]
        self.unfitted_data = None

        cached_shift_indices = self.cached_shift_indices
        self.cached_shift_indices = None
        data_num_list = self.data_num_list
        self.data_num_list = [0]
        fitted_data_num = self.fitted_data_num
        self.fitted_data_num = self.dataset.get_n_samples()

        # TODO:先还是展示当前投影的结果，然后在后台更新投影模型，模型更新完成后，再更新所有数据的投影
        must_indices = np.union1d(cached_shift_indices, np.arange(fitted_data_num, self.dataset.get_n_samples(), 1))

        self.training_queue_set.training_data_queue.put([self.finetune_epoch, fitted_data_num, data_num_list,
                                                         cur_data_num, self.pre_embeddings, must_indices])

        return cur_total_embeddings

    def model_update_final(self):
        # 暂时先等待之前的模型训练完
        initial_trainer_stat = self.training_queue_set.MODEL_UPDATING.value
        while self.training_queue_set.MODEL_UPDATING.value == 1:
            pass

        if initial_trainer_stat == 1:
            while not self.training_queue_set.embedding_queue.empty():
                self.training_queue_set.embedding_queue.get()   # 取出之前训练的结果，但是在这里是没用的了

        must_indices = np.union1d(self.cached_shift_indices, np.arange(self.fitted_data_num,
                                                                       self.dataset.get_n_samples(), 1))
        self.training_queue_set.training_data_queue.put([self.finetune_epoch, self.fitted_data_num, self.data_num_list,
                                                         self.dataset.get_n_samples(), self.pre_embeddings,
                                                         must_indices])

        data_num_when_update, infer_model = self.training_queue_set.embedding_queue.get()
        self.update_scdr(infer_model)
        ret_embeddings = self.embed_current_data()
        return ret_embeddings

    def _caching_initial_data(self, data, labels):
        self.initial_data_buffer = data if self.initial_data_buffer is None \
            else np.concatenate([self.initial_data_buffer, data], axis=0)
        if labels is not None:
            self.initial_label_buffer = labels if self.initial_label_buffer is None \
                else np.concatenate([self.initial_label_buffer, labels], axis=0)

        return self.initial_data_buffer.shape[0] >= self.initial_train_num

    def buffer_empty(self):
        return self.cached_shift_indices is None

    def clear_buffer(self, final_embed=False):
        # 更新knn graph
        self.pre_embeddings = self._adapt_distribution_change()
        return self.pre_embeddings

    def _generate_key_points(self, data):
        if self.key_data is None:
            self.key_data = KeyPointsGenerater.generate(data, self.key_data_rate)[0]
        else:
            key_data = KeyPointsGenerater.generate(data, self.key_data_rate)[0]
            self.key_data = np.concatenate([self.key_data, key_data], axis=0)

    def _initial_project(self, data, labels=None):
        self.training_queue_set.training_data_queue.put([None, self.initial_train_epoch, self.ckpt_path])
        self.training_queue_set.raw_data_queue.put([data, labels])
        # self.model_initial_time = time.time() - sta
        self.training_queue_set.INITIALIZING.value = True

    def save_model(self):
        self.training_queue_set.flag_queue.put(ModelUpdateQueueSet.SAVE)

    def infer_embeddings(self, data):
        self.infer_model.to(self.device)
        data = torch.tensor(data, dtype=torch.float).to(self.device)
        with torch.no_grad():
            self.infer_model.eval()
            data_embeddings = self.infer_model(data).cpu()

        return data_embeddings

    def embed_current_data(self):
        self.pre_embeddings = self.infer_embeddings(self.dataset.total_data).numpy()
        return self.pre_embeddings

    def print_time_cost_info(self):
        output_temp = 'Model Initialize: %.4f Model Update: %.4f Model Re-infer: %.4f \n' \
                      'kNN Calculate: %.4f kNN Update: %.4f kNN Approximate: %.4f \n' \
                      'Shift Detect: %.4f' % (self.model_initial_time, self.model_update_time,
                                              self.model_repro_time, self.knn_cal_time, self.knn_update_time,
                                              self.knn_approx_time, self.shift_detect_time)
        print(output_temp)

    def update_scdr(self, newest_model, first=False, embeddings=None):
        self.infer_model = newest_model
        self.infer_model = self.infer_model.to(self.device)

        if first:
            assert embeddings is not None
            self.pre_embeddings = embeddings
            self.model_trained = True
            self.fitted_data_num = embeddings.shape[0]

    def ending(self):
        self.training_queue_set.flag_queue.put(ModelUpdateQueueSet.STOP)
