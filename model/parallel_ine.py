from multiprocessing import Process
from threading import Thread

from sklearn.neighbors import LocalOutlierFactor

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
        self.pattern_detector.daemon = True
        self.pattern_detector.start()
        self.model_updater = None
        self._newest_embeddings = None
        self._get_new_model = False

    def _first_train(self, train_data):
        self.pre_embeddings = super()._first_train(train_data)
        self.model_updater = INEUpdater(self._model_update_queue, self._model_return_queue, self.initial_train_num,
                                        self.n_components, self.n_neighbors, self.init)
        self._pattern_data_queue.put([train_data, False, train_data, ])
        self.model_updater.daemon = True
        self.model_updater.start()

    def _incremental_embedding(self, new_data):
        # 一次只处理一个数据
        pre_data_num = self.pre_embeddings.shape[0]

        if not self._replace_model_queue.empty():
            replace_model = self._replace_model_queue.get()
            if replace_model:
                print("replace model!")

                if not self._get_new_model:
                    self._get_new_model_info()

                self.pre_embeddings[:self._newest_embeddings.shape[0]] = self._newest_embeddings
                self._get_new_model = False

        if not self._model_return_queue.empty():
            self._get_new_model_info()

        self._pattern_data_queue.put([new_data, False, self.stream_dataset.get_total_data(), ])

        knn_indices, knn_dists, dists = self._cal_new_data_kNN(new_data, include_self=False)

        new_data_prob = self._cal_new_data_probability(knn_dists.astype(np.float32, copy=False))

        initial_embedding = self._initialize_new_data_embedding(pre_data_num, knn_indices)
        # print("initial", initial_embedding)
        self.pre_embeddings = self._optimize_new_data_embedding(knn_indices, initial_embedding, new_data_prob)
        # print("after", self.pre_embeddings[-1])
        return self.pre_embeddings

    def _get_new_model_info(self):
        self._newest_embeddings = self._model_return_queue.get()
        self._get_new_model = True


class INEChangeDetector(Process):
    def __init__(self, pattern_data_queue, model_update_queue, replace_model_queue, update_thresh=50, change_thresh=200):
        Process.__init__(self, name="INEChangeDetector")
        self._pattern_data_queue = pattern_data_queue
        self._model_update_queue = model_update_queue
        self._replace_model_queue = replace_model_queue
        self._lof = None
        self._cur_change_num = 0
        self._cur_unfitted_num = 0
        self._update_thresh = update_thresh
        self._change_thresh = change_thresh

    def run(self) -> None:
        while True:
            data, stop_flag, total_data = self._pattern_data_queue.get()

            if stop_flag:
                self._model_update_queue.put([True, None])
                break

            replace_model = False
            if self._lof is None:
                self._lof = LocalOutlierFactor(n_neighbors=10, novelty=True, metric="euclidean", contamination=0.1)
                self._lof.fit(data)
            else:
                self._cur_unfitted_num += data.shape[0]
                # print(self._cur_unfitted_num)
                if self._cur_unfitted_num >= self._update_thresh:
                    self._send_update_signal(stop_flag, total_data)
                    self._cur_unfitted_num = 0

                labels = self._lof.predict(data)
                self._cur_change_num += np.count_nonzero(labels-1)

                if self._cur_change_num >= self._change_thresh:
                    replace_model = True
                    self._cur_change_num = 0
                    self._lof.fit(total_data)

            self._replace_model_queue.put(replace_model)

    def _send_update_signal(self, stop_flag, total_data):
        while not self._model_update_queue.empty():
            self._model_update_queue.get()
        self._model_update_queue.put([stop_flag, total_data])


class INEUpdater(Process, INEModel):
    def __init__(self, model_update_queue, model_return_queue, train_num, n_components, n_neighbors, init="pca"):
        Process.__init__(self, name="INEUpdater")
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
