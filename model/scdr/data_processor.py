import time
from multiprocessing import Process

import numpy as np

from utils.nn_utils import StreamingKNNSearchApprox2
from model.scdr.dependencies.scdr_utils import StreamingDataRepo
from utils.queue_set import DataProcessorQueueSet


class SCDRProcessor(Process):
    SIGNAL_GATHER_DATA = 0
    SIGNAL_UPDATE_ONLY = 1
    SIGNAL_GATHER_UPDATE = 2

    def __init__(self, n_neighbors, shift_buffer_size, data_process_queue_set, model_update_queue_set,
                 cal_time_queue_set, device):
        Process.__init__(self, name="数据处理进程")
        self.data_process_queue_set = data_process_queue_set
        self.model_update_queue_set = model_update_queue_set
        self.cal_time_queue_set = cal_time_queue_set
        self.n_neighbors = n_neighbors
        self.device = device
        self.knn_searcher_approx = StreamingKNNSearchApprox2()
        self.pre_train_num = 0
        self.unfitted_data = None
        self.cached_shift_indices = []
        self.shift_buffer_size = shift_buffer_size
        self.dataset = StreamingDataRepo(self.n_neighbors)
        self.data_num_list = [0]

    def run(self) -> None:
        while True:

            # 进程会在这里阻塞住
            fitted_num, query_embeddings, query_data, query_labels, shift_indices, SIGNAL \
                = self.data_process_queue_set.data_queue.get()

            # 用于第一次模型更新后更新pre_train_num
            if self.pre_train_num == 0:
                self.pre_train_num = fitted_num

            pre_n_samples = self.dataset.get_n_samples()
            if SIGNAL == SCDRProcessor.SIGNAL_UPDATE_ONLY:
                self._send_update_signal(pre_n_samples)
                continue

            self.dataset.add_new_data(query_data, query_embeddings, query_labels)
            if SIGNAL == SCDRProcessor.SIGNAL_GATHER_DATA:
                continue

            self.data_num_list.append(query_data.shape[0] + self.data_num_list[-1])
            fitted_embeddings = self.dataset._total_embeddings[:fitted_num]
            fitted_data = self.dataset._total_data[:fitted_num]
            sta = time.time()
            nn_indices, nn_dists = self.knn_searcher_approx.search(self.n_neighbors, fitted_embeddings, fitted_data,
                                                                   query_embeddings, query_data, self.unfitted_data)
            self.cal_time_queue_set.knn_approx_queue.put(time.time() - sta)

            # 返回kNN数据
            self.model_update_queue_set.raw_data_queue.put([query_data, nn_indices, nn_dists, query_labels])

            self.cached_shift_indices = shift_indices if self.cached_shift_indices is None else \
                np.concatenate([self.cached_shift_indices, shift_indices])
            if len(self.cached_shift_indices) <= self.shift_buffer_size:
                # TODO：如何计算这些最新数据的投影结果，以保证其在当前时间点是可信的，应该被当作异常点/离群点进行处理
                # TODO: 这边即使数据的分布没有发生变化，但是在投影质量下降到一定程度后，也要进行模型更新
                self.unfitted_data = query_data if self.unfitted_data is None else np.concatenate(
                    [self.unfitted_data, query_data], axis=0)
            else:
                self._send_update_signal(pre_n_samples)

    def _send_update_signal(self, pre_n_samples):
        # 如果还有模型更新任务没有完成，此处应该阻塞等待
        while self.model_update_queue_set.MODEL_UPDATING.value:
            pass
        self.model_update_queue_set.MODEL_UPDATING.value = True
        fitted_data_num = self.pre_train_num

        self.unfitted_data = None
        cached_shift_indices = self.cached_shift_indices
        self.cached_shift_indices = []
        data_num_list = self.data_num_list
        self.data_num_list = [0]
        # TODO:先还是展示当前投影的结果，然后在后台更新投影模型，模型更新完成后，再更新所有数据的投影
        must_indices = np.union1d(cached_shift_indices,
                                  np.arange(fitted_data_num, self.dataset.get_n_samples(), 1))

        self.pre_train_num = self.dataset.get_n_samples()
        self.model_update_queue_set.training_data_queue.put(
            [fitted_data_num, data_num_list, self.pre_train_num,
             self.dataset._total_embeddings[:pre_n_samples], must_indices])

