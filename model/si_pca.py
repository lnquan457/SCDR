import copy
import time
from threading import Thread

import numpy as np
from inc_pca import IncPCA
from sklearn.neighbors import LocalOutlierFactor


class StreamingIPCA:
    def __init__(self, n_components, forgetting_factor=1.0):
        self.n_components = n_components
        self.pca_model = IncPCA(n_components, forgetting_factor)
        self.total_data = None
        self.pre_embeddings = None
        self.time_cost = 0
        self._time_cost_records = [0]

    def fit_new_data(self, x, labels=None):
        if self.total_data is None:
            self.total_data = x
        else:
            self.total_data = np.concatenate([self.total_data, x], axis=0)
        sta = time.time()
        self.pca_model.partial_fit(x)
        cur_embeddings = self.pca_model.transform(self.total_data)

        if self.pre_embeddings is None:
            self.pre_embeddings = cur_embeddings
        else:
            self.pre_embeddings = IncPCA.geom_trans(self.pre_embeddings, cur_embeddings)
        self.time_cost += time.time() - sta
        self._time_cost_records.append(time.time() - sta + self._time_cost_records[-1])
        return self.pre_embeddings

    def ending(self):
        output = "Time Cost: %.4f" % self.time_cost
        print(output)
        return output, self._time_cost_records


class ParallelsPCA:
    def __init__(self, pattern_data_queue, model_update_queue, model_return_queue, replace_model_queue, n_components,
                 forgetting_factor=1.0):
        self.n_components = n_components
        self._forgetting_factor = forgetting_factor
        self.pca_model = None
        self.total_data = None
        self.pre_embeddings = None
        self._newest_model = None
        self._pattern_data_queue = pattern_data_queue
        self._model_return_queue = model_return_queue
        self._replace_model_queue = replace_model_queue
        self._model_update_queue = model_update_queue

        self.pattern_detector = PCAPatternChangeDetector(pattern_data_queue, model_update_queue, replace_model_queue)
        self.model_updater = None

        self.time_cost = 0
        self._time_cost_records = [0]

    def fit_new_data(self, x, labels=None):
        replace_model = False
        if self.total_data is None:
            self.total_data = x
            self.model_updater = PCAUpdater(self._model_update_queue, self._model_return_queue, self.n_components,
                                            self._forgetting_factor)
            self._model_update_queue.put([False, x])
            self.model_updater.start()
            self.pca_model = self._model_return_queue.get()
            self.pattern_detector.start()
            self._pattern_data_queue.put([self.total_data, False, self.total_data, ])
            replace_model = True
        else:
            self.total_data = np.concatenate([self.total_data, x], axis=0)

        sta = time.time()

        if not self._replace_model_queue.empty():
            replace_model = self._replace_model_queue.get()
            if replace_model:
                print("replace model!")
                self.pca_model = self._newest_model

        if not self._model_return_queue.empty():
            self._newest_model = self._model_return_queue.get()

        self._pattern_data_queue.put([x, False, self.total_data, ])

        if replace_model:
            cur_embeddings = self.pca_model.transform(self.total_data)
            if self.pre_embeddings is None:
                self.pre_embeddings = cur_embeddings
            else:
                self.pre_embeddings = IncPCA.geom_trans(self.pre_embeddings, cur_embeddings)
        else:
            cur_embeddings = self.pca_model.transform(x)
            self.pre_embeddings = np.concatenate([self.pre_embeddings, cur_embeddings], axis=0)

        self.time_cost += time.time() - sta
        self._time_cost_records.append(time.time() - sta + self._time_cost_records[-1])
        return self.pre_embeddings

    def ending(self):
        self._pattern_data_queue.put([None, True, self.total_data])
        output = "Time Cost: %.4f" % self.time_cost
        print(output)
        return output, self._time_cost_records


class PCAPatternChangeDetector(Thread):
    def __init__(self, pattern_data_queue, model_update_queue, replace_model_queue, update_thresh=50, change_thresh=200):
        Thread.__init__(self, name="PatternChangeDetector")
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
        self._model_update_queue.put([stop_flag, total_data[-self._cur_unfitted_num:]])


class PCAUpdater(Thread):
    def __init__(self, model_update_queue, model_return_queue, n_components, forgetting_factor):
        Thread.__init__(self, name="PCAUpdater")
        self._model = None
        self._n_components = n_components
        self._forgetting_factor = forgetting_factor
        self._model_update_queue = model_update_queue
        self._model_return_queue = model_return_queue

    def run(self) -> None:
        while True:
            stop_flag, data = self._model_update_queue.get()

            if stop_flag:
                break

            if self._model is None:
                self._model = IncPCA(self._n_components, self._forgetting_factor)

            self._model.partial_fit(data)
            self._model_return_queue.put(copy.copy(self._model))
