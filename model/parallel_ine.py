from threading import Thread

from model.ine import INEModel
from model.si_pca import PCAPatternChangeDetector
import numpy as np


class ParallelINE(INEModel):
    def __init__(self, pattern_data_queue, model_update_queue, model_return_queue, replace_model_queue,
                 train_num, n_components, n_neighbors, iter_num=100, grid_num=27, desired_perplexity=3, init="random"):
        INEModel.__init__(self, train_num, n_components, n_neighbors, iter_num, grid_num, desired_perplexity, init)
        self._pattern_data_queue = pattern_data_queue
        self._model_return_queue = model_return_queue
        self._replace_model_queue = replace_model_queue
        self._model_update_queue = model_update_queue
        self.pattern_detector = INEChangeDetector(pattern_data_queue, model_update_queue, replace_model_queue)
        self.model_updater = None
        self._newest_embeddings = None

    def _first_train(self, train_data):
        self.pre_embeddings = super()._first_train(train_data)
        self.model_updater = INEUpdater(self._model_update_queue, self._model_return_queue, self.initial_train_num,
                                        self.n_components, self.n_neighbors, self.init)
        self.pattern_detector.start()
        self._pattern_data_queue.put([train_data, False, train_data, ])
        self.model_updater.start()

    def _incremental_embedding(self, new_data):
        # 一次只处理一个数据
        new_data = np.reshape(new_data, (1, -1))
        pre_data_num = self.knn_manager.knn_indices.shape[0]

        if not self._replace_model_queue.empty():
            replace_model = self._replace_model_queue.get()
            if replace_model:
                print("replace model!")
                self.pre_embeddings[:self._newest_embeddings.shape[0]] = self._newest_embeddings

        if not self._model_return_queue.empty():
            self._newest_embeddings = self._model_return_queue.get()

        self._pattern_data_queue.put([new_data, False, self.stream_dataset.get_total_data(), ])

        knn_indices, knn_dists, dists = self._cal_new_data_kNN(new_data, include_self=False)

        new_data_prob = self._cal_new_data_probability(knn_dists.astype(np.float32, copy=False))

        initial_embedding = self._initialize_new_data_embedding(pre_data_num, knn_indices)
        # print("initial", initial_embedding)
        self.pre_embeddings = self._optimize_new_data_embedding(knn_indices, initial_embedding, new_data_prob)
        # print("after", self.pre_embeddings[-1])
        return self.pre_embeddings


class INEChangeDetector(PCAPatternChangeDetector):
    def _send_update_signal(self, stop_flag, total_data):
        self._model_update_queue.put([stop_flag, total_data])


class INEUpdater(Thread, INEModel):
    def __init__(self, model_update_queue, model_return_queue, train_num, n_components, n_neighbors, init="pca"):
        Thread.__init__(self, name="INEUpdater")
        INEModel.__init__(self, train_num, n_components, n_neighbors, init)
        self._n_components = n_components
        self._n_neighbors = n_neighbors
        self._init = init
        self._model_update_queue = model_update_queue
        self._model_return_queue = model_return_queue

    def run(self) -> None:
        while True:
            stop_flag, data = self._model_update_queue.get()

            if stop_flag:
                break

            embeddings = self.fit_transform(data)
            self._model_return_queue.put(embeddings)
