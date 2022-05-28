import time

import numpy as np

from experiments.scdr_trainer import SCDRTrainer
from model.scdr import SCDRBase


class RTSCDRModel(SCDRBase):
    def __init__(self, shift_buffer_size, n_neighbors, model_trainer: SCDRTrainer, initial_train_num,
                 initial_train_epoch, finetune_epoch, finetune_data_ratio=1.0, ckpt_path=None):
        SCDRBase.__init__(self, n_neighbors, model_trainer, initial_train_num, initial_train_epoch, finetune_epoch,
                          finetune_data_ratio, ckpt_path)

        self.shift_buffer_size = shift_buffer_size
        self.initial_data_buffer = None
        self.initial_label_buffer = None

    def fit_new_data(self, data, labels=None):
        if not self.model_trained:
            if not self._caching_initial_data(data, labels):
                return None
            self._initial_project(self.initial_data_buffer, self.initial_label_buffer)
            sta = time.time()
            self.knn_searcher.search(self.initial_data_buffer, self.n_neighbors, just_add_new_data=True)
            self.knn_cal_time += time.time() - sta
        else:
            shifted_indices = self._detect_distribution_shift(data, labels)
            self.data_num_list.append(data.shape[0] + self.data_num_list[-1])

            # 更新knn graph
            sta = time.time()
            nn_indices, nn_dists = self.knn_searcher.search(data, self.n_neighbors)
            self.knn_cal_time += time.time() - sta
            self.dataset.add_new_data(data, nn_indices, nn_dists, labels)

            if len(self.cached_shift_indices) <= self.shift_buffer_size:
                # TODO：如何计算这些最新数据的投影结果，以保证其在当前时间点是可信的，应该被当作异常点/离群点进行处理
                sta = time.time()
                data_embeddings = self.model_trainer.infer_embeddings(data)
                data_embeddings = np.reshape(data_embeddings, (data.shape[0], self.pre_embeddings.shape[1]))
                self.model_repro_time += time.time() - sta
                self.pre_embeddings = np.concatenate([self.pre_embeddings, data_embeddings], axis=0)
            else:
                self._process_shift_situation()
                self.cached_shift_indices = None
        # self._gather_data_stream()
        return self.pre_embeddings

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
        self._process_shift_situation()
        return self.pre_embeddings

