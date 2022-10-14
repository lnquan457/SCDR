import os.path
import time

import numpy as np
import torch
from model.scdr.data_processor import SCDRProcessor
from utils.queue_set import ModelUpdateQueueSet, DataProcessorQueueSet
from model.scdr.dependencies.scdr_utils import KeyPointsGenerator, DistributionChangeDetector


class SCDRParallel:
    def __init__(self, model_update_queue_set, data_process_queue_set, cal_time_queue_set, initial_train_num,
                 initial_train_epoch, ckpt_path=None, device="cuda:0"):
        self.model_update_queue_set = model_update_queue_set
        self.data_process_queue_set = data_process_queue_set
        self.cal_time_queue_set = cal_time_queue_set
        self.device = device
        self.infer_model = None
        self.ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
        self.initial_train_num = initial_train_num
        self.initial_train_epoch = initial_train_epoch

        self.initial_data_buffer = None
        self.initial_label_buffer = None
        self.pre_embeddings = None

        self.time_step = 0
        self.model_trained = False
        # 用户告知shift detector是否需要重新fit已经见过的数据
        self.model_just_updated = False
        self.fitted_data_num = 0

        self.key_data = None
        self.key_indices = None
        # self.key_data_rate = 0.5
        self.key_data_rate = 1.0

        # 没有经过模型拟合的数据
        self.unfitted_data = None
        self.total_data = None
        self.total_labels = None

        self.dis_change_detector = DistributionChangeDetector(lof_based=True)

        self.shift_detect_time = 0
        self.model_repro_time = 0
        self.debug = True
        self.update_when_end = True

    # scdr本身属于嵌入进程，只负责数据的嵌入
    def fit_new_data(self, data, labels=None):
        new_data_num = data.shape[0]
        # 这一步竟然也还挺耗时的，不应该每次有数据来的时候就进行采样。因为key_data只有在shift detect的时候用于进行refit，
        # 所以只需要在每次模型更新之后再进行重新采样就可以了。
        if self.model_just_updated:
            sta = time.time()
            self._generate_key_points(self.total_data[:self.fitted_data_num])
            print("generate time", time.time() - sta)
        self.total_data = data if self.total_data is None else np.concatenate([self.total_data, data], axis=0)
        self.total_labels = labels if self.total_labels is None else np.concatenate([self.total_labels, labels])

        # 模型第一次训练过程中，model_trained状态没有切换，此时接收到的数据还是在cache当中的？ 没有，数据进程被阻塞住了
        if not self.model_trained:
            if not self._caching_initial_data(data, labels):
                return None

            self.data_process_queue_set.data_queue.put([self.fitted_data_num, None, self.initial_data_buffer,
                                                        self.initial_label_buffer, None,
                                                        SCDRProcessor.SIGNAL_GATHER_DATA])

            self._initial_project(self.initial_data_buffer, self.initial_label_buffer)

        else:
            sta = time.time()
            shifted_indices = self.detect_distribution_change(data, labels)
            self.shift_detect_time += time.time() - sta

            # 使用之前的投影函数对最新的数据进行投影
            sta = time.time()

            # 后续这里要根据数据的分布情况实施不同的嵌入策略
            data_embeddings = self.infer_embeddings(data).numpy().astype(float)
            self.model_repro_time += time.time() - sta
            data_embeddings = np.reshape(data_embeddings, (new_data_num, self.pre_embeddings.shape[1]))

            self.pre_embeddings = np.concatenate([self.pre_embeddings, data_embeddings], axis=0, dtype=float)

            self.data_process_queue_set.data_queue.put(
                [self.fitted_data_num, data_embeddings, data, labels, shifted_indices, SCDRProcessor.SIGNAL_GATHER_UPDATE])

        return self.pre_embeddings

    def detect_distribution_change(self, data, labels=None):
        # 这可能会导致分布变化检测的精度降低，从而增加模型更新的频率
        # 这样处理了之后，模型更新的速度完全跟不上投影的速度，新分布的数据到达时模型还没有拟合之前的数据，就导致不断的触发模型更新。
        if self.model_just_updated:
            refit = True
            fit_data = self.key_data
            self.model_just_updated = False
        else:
            fit_data = None
            refit = False
        shifted_indices = self.dis_change_detector.detect_distribution_shift(data, refit, fit_data,
                                                                             self.total_labels[self.key_indices], labels)
        shifted_indices = shifted_indices + (self.total_data.shape[0] - data.shape[0])
        return shifted_indices

    def get_final_embeddings(self):
        if self.update_when_end:
            # print("final update")
            # 暂时先等待之前的模型训练完
            while self.model_update_queue_set.MODEL_UPDATING.value == 1:
                pass

            if not self.model_update_queue_set.embedding_queue.empty():
                self.model_update_queue_set.embedding_queue.get()  # 取出之前训练的结果，但是在这里是没用的了

            self.data_process_queue_set.data_queue.put(
                [self.fitted_data_num, None, None, None, None, SCDRProcessor.SIGNAL_UPDATE_ONLY])

            data_num_when_update, infer_model = self.model_update_queue_set.embedding_queue.get()
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

    def _generate_key_points(self, data):
        # if self.key_data is None:
        self.key_data, self.key_indices = KeyPointsGenerator.generate(data, self.key_data_rate)
        # else:
        #     key_data, key_indices = KeyPointsGenerater.generate(data, self.key_data_rate)
        #     self.key_data = np.concatenate([self.key_data, key_data], axis=0)
        #     self.key_indices = np.concatenate([self.key_indices, key_indices])

    def _initial_project(self, data, labels=None):
        self.model_update_queue_set.training_data_queue.put([None, self.initial_train_epoch, self.ckpt_path])
        self.model_update_queue_set.raw_data_queue.put([data, labels])
        # self.model_initial_time = time.time() - sta
        self.model_update_queue_set.INITIALIZING.value = True

    def save_model(self):
        self.model_update_queue_set.flag_queue.put(ModelUpdateQueueSet.SAVE)

    def infer_embeddings(self, data):
        self.infer_model.to(self.device)
        data = torch.tensor(data, dtype=torch.float).to(self.device)
        with torch.no_grad():
            self.infer_model.eval()
            data_embeddings = self.infer_model(data).cpu()

        return data_embeddings

    def embed_current_data(self):
        self.pre_embeddings = self.infer_embeddings(self.total_data).numpy()
        return self.pre_embeddings

    def update_scdr(self, newest_model, first=False, embeddings=None, embedding_num=None):
        self.infer_model = newest_model
        self.infer_model = self.infer_model.to(self.device)
        self.fitted_data_num = embedding_num
        self.model_just_updated = True
        if first:
            assert embeddings is not None
            self.pre_embeddings = embeddings
            self.model_trained = True

            self.data_process_queue_set.data_queue.put([self.fitted_data_num, embeddings, None, None, None,
                                                        SCDRProcessor.SIGNAL_GATHER_DATA])

    def ending(self):
        self.dis_change_detector.ending()
        def accumulate(queue_obj):
            total = 0
            while not queue_obj.empty():
                total += queue_obj.get()
            return total

        knn_approx = accumulate(self.cal_time_queue_set.knn_approx_queue)
        knn_update = accumulate(self.cal_time_queue_set.knn_update_queue)
        model_update = accumulate(self.cal_time_queue_set.model_update_queue)
        model_initial = accumulate(self.cal_time_queue_set.model_initial_queue)
        training_data_sample = accumulate(self.cal_time_queue_set.training_data_sample_queue)
        print("Shift Detect: %.4f kNN Approx: %.4f kNN Update: %.4f Model Update: %.4f Model Initial: %.4f "
              "Model Refer: %.4f Data Sample: %.4f" % (self.shift_detect_time, knn_approx, knn_update, model_update,
                                                       model_initial, self.model_repro_time, training_data_sample))
