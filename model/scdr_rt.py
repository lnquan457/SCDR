import time

import numpy as np

from experiments.scdr_trainer import SCDRTrainer
from model.scdr import SCDRBase
from utils.scdr_utils import KeyPointsGenerater


class RTSCDRModel(SCDRBase):
    def __init__(self, shift_buffer_size, n_neighbors, model_trainer: SCDRTrainer, initial_train_num,
                 initial_train_epoch, finetune_epoch, finetune_data_ratio=0.5, ckpt_path=None):
        SCDRBase.__init__(self, n_neighbors, model_trainer, initial_train_num, initial_train_epoch, finetune_epoch,
                          finetune_data_ratio, ckpt_path)

        self.shift_buffer_size = shift_buffer_size
        self.initial_data_buffer = None
        self.initial_label_buffer = None

        # 这种情况会导致算法检测不准确，所以检测出异常点的可能性更高（因为高维分布的数据少了），
        # 会导致投影函数更新的次数增加。
        self.key_data_rate = 0.5
        self.using_key_data_in_lof = True

        self.using_key_data_in_finetune = True
        self.sample_prob = None
        self.min_prob = 0.1
        self.decay_rate = 0.9
        # TODO：后续还需要模拟真实的流数据环境，数据的产生和数据的处理应该是两个进程并行的
        # 采样概率衰减的间隔时间，以秒为单位
        self.pre_update_time = None
        self.decay_iter_time = 10

    def fit_new_data(self, data, labels=None):
        if self.using_key_data_in_lof:
            self._generate_key_points(data)

        if self.using_key_data_in_finetune:
            # 如果速率产生的很快，那么采样的概率也会衰减的很快，所以这里的更新应该要以时间为标准
            self._update_sample_prob(data.shape[0])

        if not self.model_trained:
            if not self._caching_initial_data(data, labels):
                return None
            self._initial_project(self.initial_data_buffer, self.initial_label_buffer)
            sta = time.time()
            self.knn_searcher.search(self.initial_data_buffer, self.n_neighbors, just_add_new_data=True)
            self.knn_cal_time += time.time() - sta
        else:
            fit_data = self.key_data if self.using_key_data_in_lof else self.dataset.total_data
            shifted_indices = self._detect_distribution_shift(fit_data, data, labels)
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

    def _update_sample_prob(self, n_samples):
        # 之后还可以考虑其他一些指标一起计算采样概率
        if self.sample_prob is None:
            self.sample_prob = np.ones(shape=n_samples)
            self.pre_update_time = time.time()
        else:
            cur_time = time.time()
            if cur_time - self.pre_update_time > self.decay_iter_time:
                self.sample_prob *= self.decay_rate
                self.sample_prob[self.sample_prob < self.min_prob] = self.min_prob
                self.pre_update_time = cur_time
            self.sample_prob = np.concatenate([self.sample_prob, np.ones(shape=n_samples)])

    def _sample_training_data(self):
        # 采样训练子集
        if not self.using_key_data_in_finetune:
            return super()._sample_training_data()
        else:
            sampled_indices = KeyPointsGenerater.generate(self.dataset.total_data, self.finetune_data_rate,
                                                          prob=self.sample_prob)[1]
            if not self.buffer_empty():
                sampled_indices = np.union1d(sampled_indices, self.cached_shift_indices)
            return sampled_indices

    def _generate_key_points(self, data):
        if self.key_data is None:
            self.key_data = KeyPointsGenerater.generate(data, self.key_data_rate)[0]
        else:
            key_data = KeyPointsGenerater.generate(data, self.key_data_rate)[0]
            self.key_data = np.concatenate([self.key_data, key_data], axis=0)
