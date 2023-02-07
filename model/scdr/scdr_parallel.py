import os.path
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from dataset.warppers import StreamingDatasetWrapper
from model.scdr.data_processor import DataProcessor, DataProcessorProcess
from model.scdr.dependencies.embedding_optimizer import NNEmbedder
from utils.queue_set import ModelUpdateQueueSet, DataProcessorQueue


class SCDRParallel:
    def __init__(self, n_neighbors, batch_size, model_update_queue_set, initial_train_num, window_size, ckpt_path=None,
                 device="cuda:0"):
        self.model_update_queue_set = model_update_queue_set
        self.device = device
        self.nn_embedder = NNEmbedder(device)
        self.data_processor = DataProcessor(n_neighbors, batch_size, model_update_queue_set, device)
        self.ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
        self.initial_train_num = initial_train_num

        # 进行初次训练后，初始化以下对象
        self.stream_dataset = StreamingDatasetWrapper(batch_size, n_neighbors, window_size, self.device)

        self.initial_data_buffer = None
        self.initial_label_buffer = None

        self.model_trained = False

        self.model_infer_time = 0
        self._time_cost_records = [0]

    # scdr本身属于嵌入进程，负责数据的处理和嵌入
    def fit_new_data(self, data, labels=None, end=False):
        # w_sta = time.time()
        other_time = 0
        if not self.model_trained:
            if not self._caching_initial_data(data, labels):
                return None, 0, False

            self.stream_dataset.add_new_data(data=self.initial_data_buffer, labels=self.initial_label_buffer)
            total_embeddings = self._initial_project_model()
            self.data_processor.init_stream_dataset(self.stream_dataset)
            self.data_processor.nn_embedder = self.nn_embedder
        else:
            if data is None:
                return None, 0, False
            self._listen_model_update()

            sta = time.time()
            data_embeddings = self.nn_embedder.embed(data)
            self.model_infer_time += time.time() - sta
            total_embeddings, other_time = self.data_processor.process(data, data_embeddings, labels)
            self._time_cost_records.append(time.time() - sta - other_time + self._time_cost_records[-1])

        return total_embeddings, other_time, True

    def _listen_model_update(self):
        if not self.model_update_queue_set.embedding_queue.empty():
            embeddings, infer_model, stream_dataset, total_data_idx = self.model_update_queue_set.embedding_queue.get()
            self.update_scdr(infer_model, embeddings, stream_dataset, total_data_idx)

    def get_final_embeddings(self):
        return self.data_processor.get_final_embeddings()

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
            [self.stream_dataset, None, self.ckpt_path])
        self.model_update_queue_set.flag_queue.put(ModelUpdateQueueSet.UPDATE)
        self.model_update_queue_set.INITIALIZING.value = True
        # 这里传回来的stream_dataset中，包含了最新的对称kNN信息
        embeddings, model, stream_dataset, _ = self.model_update_queue_set.embedding_queue.get()
        self.stream_dataset = stream_dataset
        self.stream_dataset.update_unfitted_data_num(0)
        self.stream_dataset.add_new_data(embeddings=embeddings)
        self.nn_embedder.update_model(model)

        self.model_trained = True
        self.model_update_queue_set.INITIALIZING.value = False

        return embeddings

    def save_model(self):
        self.model_update_queue_set.flag_queue.put(ModelUpdateQueueSet.SAVE)

    def update_scdr(self, newest_model, embeddings, stream_dataset, total_data_idx):
        return self.data_processor.update_scdr(newest_model, embeddings, stream_dataset, total_data_idx)

    def ending(self):
        print("Model Infer: %.4f " % self.model_infer_time)

        return self.data_processor.ending(), self._time_cost_records


class SCDRFullParallel(SCDRParallel):
    def __init__(self, embedding_data_queue, n_neighbors, batch_size, model_update_queue_set, initial_train_num,
                 ckpt_path=None, device="cuda:0", window_size=2000):
        SCDRParallel.__init__(self, n_neighbors, batch_size, model_update_queue_set, initial_train_num, window_size,
                              ckpt_path, device)
        self.data_processor = DataProcessorProcess(embedding_data_queue, n_neighbors, batch_size, model_update_queue_set
                                                   , window_size, device)
        self._embedding_data_queue: DataProcessorQueue = embedding_data_queue

    def fit_new_data(self, data, labels=None, end=False):
        if not self.model_trained:

            if not self._caching_initial_data(data, labels):
                return None, 0, False

            self.stream_dataset.add_new_data(data=self.initial_data_buffer, labels=self.initial_label_buffer)
            total_embeddings = self._initial_project_model()
            self.data_processor.init_stream_dataset(self.stream_dataset)

            self.data_processor.daemon = True
            self.data_processor.start()
            self._embedding_data_queue.put_res([total_embeddings, None, 0, 0])
            add_data_time = 0
        else:
            if data is None:
                self._embedding_data_queue.put([data, None, labels, end])
                return None, 0, False

            data_embeddings = self.nn_embedder.embed(data)

            total_embeddings, newest_model, cost_time, add_data_time = self._embedding_data_queue.get_res()
            sta = time.time()
            total_embeddings = np.concatenate([total_embeddings, data_embeddings], axis=0)

            if newest_model is not None:
                self.nn_embedder.update_model(newest_model)
            self._embedding_data_queue.put([data, data_embeddings, labels, end])
            self._time_cost_records.append(time.time() - sta + cost_time + self._time_cost_records[-1])

        return total_embeddings, add_data_time, True


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
