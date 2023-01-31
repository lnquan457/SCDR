import time
from copy import copy
from multiprocessing import Process
from threading import Thread

from scipy.spatial.distance import cdist
from sklearn.neighbors import LocalOutlierFactor

from model.dr_models.upd import UPDis4Streaming
from model.si_pca import PCAPatternChangeDetector
from model.xtreaming import sampled_control_points, procrustes_analysis, XtreamingModel

import numpy as np


class ParallelXtreaming:
    def __init__(self, pattern_data_queue, model_return_queue, replace_model_queue, model_update_queue,
                 buffer_size=1000, eta=0.99, window_size=1000):

        self._buffer_size = buffer_size
        self._eta = eta
        self._window_size = window_size

        self.pro_model = None
        self.buffered_data = None
        self.total_data = None
        self.pre_embeddings = None
        self._pre_control_indices = None
        self._pre_control_points = None
        self._get_new_model = False
        self._newest_model = None
        self._newest_cntp_indices = None
        self._newest_cntp_points = None
        self._pattern_data_queue = pattern_data_queue
        self._model_return_queue = model_return_queue
        self._replace_model_queue = replace_model_queue
        self._model_update_queue = model_update_queue

        self.pattern_detector = XtreamingChangeDetector(pattern_data_queue, model_update_queue, replace_model_queue)
        self.model_updater = None

        self.time_cost = 0
        self._time_cost_records = [0]
        self._total_n_samples = 0

    def _buffering(self, data):
        if self.buffered_data is None:
            self.buffered_data = data
        else:
            self.buffered_data = np.concatenate([self.buffered_data, data], axis=0)

        return self.buffered_data.shape[0] >= self._buffer_size

    def fit_new_data(self, x, labels=None):

        if not self._buffering(x):
            return None

        self._total_n_samples += self.buffered_data.shape[0]
        out_num = self._total_n_samples - self._window_size
        if out_num > 0:
            self.pre_embeddings = self.pre_embeddings[out_num:]
            self.total_data = self.total_data[out_num:]
            self._total_n_samples = self.pre_embeddings.shape[0]
            self._pre_control_indices -= out_num
            valid_indices = np.argwhere(self._pre_control_indices >= 0).squeeze()
            self._pre_control_indices = self._pre_control_indices[valid_indices].squeeze()
            self._pre_control_points = self._pre_control_points[valid_indices].squeeze()

            self.pro_model.control_points = self.pro_model.control_points[valid_indices]
            self.pro_model.ctp2ctp_dists = self.pro_model.ctp2ctp_dists[valid_indices, :][:, valid_indices]
            self.pro_model.control_embeddings = self.pro_model.control_embeddings[valid_indices]
            self.pro_model.normed_eigen_vector = self.pro_model.normed_eigen_vector[:, valid_indices]

        centroids, sampled_indices = sampled_control_points(self.buffered_data)
        replace_model = False
        if self.total_data is None:
            self.total_data = self.buffered_data
            self.model_updater = XtreamingUpdater(self._model_update_queue, self._model_return_queue, self._eta)

            self._model_update_queue.put([False, self.buffered_data, centroids, sampled_indices, 0])
            self.model_updater.start()
            self.pro_model, self.pre_embeddings, _, _ = self._model_return_queue.get()
            self._pre_control_indices = sampled_indices
            self._pre_control_points = centroids
            self.pattern_detector.start()
            self._pattern_data_queue.put([x, False, self.total_data, self._pre_control_indices, 0])
            self._newest_cntp_indices = sampled_indices
            self.buffered_data = None
            return self.pre_embeddings, 0

        sta = time.time()
        self.total_data = np.concatenate([self.total_data, self.buffered_data], axis=0)
        add_data_time = time.time() - sta

        sta = time.time()

        if not self._replace_model_queue.empty():
            replace_model = self._replace_model_queue.get()
            if replace_model:
                print("replace model!")

                if not self._get_new_model:
                    self._get_new_model_info()

                self.pro_model = self._newest_model
                self._pre_control_indices = self._newest_cntp_indices
                self._pre_control_points = self._newest_cntp_points
                self._get_new_model = False

        if not self._model_return_queue.empty():
            self._get_new_model_info()

        self._pattern_data_queue.put([x, False, self.total_data, self._pre_control_indices, out_num])

        if replace_model:
            cur_embeddings = self.transform(self.total_data)

            if self.pre_embeddings is None:
                self.pre_embeddings = cur_embeddings
            else:
                self.pre_embeddings = procrustes_analysis(self.pre_embeddings[self._pre_control_indices],
                                                          cur_embeddings[self._pre_control_indices],
                                                          cur_embeddings, align=True)
        else:
            cur_embeddings = self.transform(x)
            self.pre_embeddings = np.concatenate([self.pre_embeddings, cur_embeddings], axis=0)

        self.time_cost += time.time() - sta
        self._time_cost_records.append(time.time() - sta - add_data_time + self._time_cost_records[-1])
        self.buffered_data = None
        return self.pre_embeddings, add_data_time

    def _get_new_model_info(self):
        self._newest_model, _, self._newest_cntp_indices, self._newest_cntp_points = self._model_return_queue.get()
        self._get_new_model = True

    def transform(self, data):
        dists = cdist(data, self.total_data[self._pre_control_indices])
        embeddings = self.pro_model.reuse_project(dists)
        return embeddings


class XtreamingChangeDetector(Process):
    def __init__(self, pattern_data_queue, model_update_queue, replace_model_queue, update_thresh=50,
                 change_thresh=200):
        Process.__init__(self, name="PatternChangeDetector")
        self._pattern_data_queue = pattern_data_queue
        self._model_update_queue = model_update_queue
        self._replace_model_queue = replace_model_queue
        self._lof = None
        self._cur_change_num = 0
        self._cur_unfitted_num = 0
        self._update_thresh = update_thresh
        self._change_thresh = change_thresh
        self._total_cntp_indices = None
        self._total_out_num = 0

    def run(self) -> None:
        while True:
            data, stop_flag, total_data, control_indices, out_num = self._pattern_data_queue.get()

            if stop_flag:
                self._model_update_queue.put([True, None])
                break

            if out_num > 0 and self._total_cntp_indices is not None:
                self._total_cntp_indices -= out_num
                valid_indices = np.argwhere(self._total_cntp_indices >= 0).squeeze()
                self._total_cntp_indices = self._total_cntp_indices[valid_indices]
                self._total_out_num += out_num

            replace_model = False
            if self._lof is None:
                self._lof = LocalOutlierFactor(n_neighbors=10, novelty=True, metric="euclidean", contamination=0.1)
                self._lof.fit(total_data[control_indices])
                self._total_cntp_indices = control_indices
            else:
                self._cur_unfitted_num += data.shape[0]
                if self._cur_unfitted_num >= self._update_thresh:
                    unfitted_data = total_data[-self._cur_unfitted_num:]
                    unfitted_cntp_points, unfitted_cntp_indices = sampled_control_points(unfitted_data)
                    self._total_cntp_indices = np.concatenate([self._total_cntp_indices, unfitted_cntp_indices +
                                                               total_data.shape[0] - self._cur_unfitted_num], axis=0)

                    self._model_update_queue.put([stop_flag, unfitted_data, unfitted_cntp_points,
                                                  unfitted_cntp_indices, self._total_out_num])
                    self._cur_unfitted_num = 0
                    self._total_out_num = 0

                labels = self._lof.predict(data)
                self._cur_change_num += np.count_nonzero(labels-1)
                # print("self._cur_change_num", self._cur_change_num)

                if self._cur_change_num >= self._change_thresh:
                    replace_model = True
                    self._cur_change_num = 0
                    self._lof.fit(total_data[self._total_cntp_indices])

            self._replace_model_queue.put(replace_model)


class XtreamingUpdater(Process):
    def __init__(self, model_update_queue, model_return_queue, eta):
        Process.__init__(self, name="XtreamingUpdater")
        self._model = None
        self._eta = eta
        self._model_update_queue = model_update_queue
        self._model_return_queue = model_return_queue
        self._pre_control_points = None
        self._pre_control_indices = None
        self._pre_embedding = None
        self._pre_cntp_embeddings = None

    def run(self) -> None:
        while True:
            stop_flag, data, cntp_points, cntp_indices, total_out_num = self._model_update_queue.get()

            if stop_flag:
                break

            if self._model is None:
                self._model = UPDis4Streaming(self._eta)
                dists = cdist(data, data)
                cntp2cntp_dists = dists[cntp_indices, :][:, cntp_indices]
                dists2cntp = dists[:, cntp_indices]
                self._pre_control_points = cntp_points
                self._pre_control_indices = cntp_indices
                self._pre_embedding = self._model.fit_transform(dists2cntp, cntp2cntp_dists)
                self._pre_cntp_embeddings = self._pre_embedding[cntp_indices]
            else:

                if total_out_num > 0:
                    self._pre_embedding = self._pre_embedding[total_out_num:]
                    self._pre_control_indices -= total_out_num
                    valid_indices = np.argwhere(self._pre_control_indices >= 0).squeeze()
                    self._pre_control_indices = self._pre_control_indices[valid_indices]
                    self._pre_control_points = self._pre_control_points[valid_indices]
                    self._pre_cntp_embeddings = self._pre_cntp_embeddings[valid_indices]

                    self._model.control_points = self._model.control_points[valid_indices]
                    self._model.ctp2ctp_dists = self._model.ctp2ctp_dists[valid_indices][:, valid_indices]
                    self._model.control_embeddings = self._model.control_embeddings[valid_indices]
                    self._model.normed_eigen_vector = self._model.normed_eigen_vector[:, valid_indices]

                pre_data_num = self._pre_embedding.shape[0]
                dists2cntp = cdist(data, self._pre_control_points)
                total_cntp_points = np.concatenate([self._pre_control_points, cntp_points], axis=0)
                # 求出之前的所有数据到之前的控制点的距离
                dists2new_cntp_1 = cdist(self._pre_embedding, self._pre_cntp_embeddings)
                # 求出之前的所有数据到当前控制点的距离
                cur_cntp_embeddings = self._model.reuse_project(dists2cntp[cntp_indices])
                dists2new_cntp_2 = cdist(self._pre_embedding, cur_cntp_embeddings)
                dists2new_cntp_all = np.concatenate([dists2new_cntp_1, dists2new_cntp_2], axis=1)

                # 所有control points之间的距离，用于构造投影函数
                total_cntp_embeddings = np.concatenate([self._pre_cntp_embeddings, cur_cntp_embeddings], axis=0)
                cntp_dists = cdist(total_cntp_embeddings, total_cntp_embeddings)
                # 对之前的数据以及所有control points进行重新投影
                prev_updated_embeddings = self._model.fit_transform(dists2new_cntp_all, cntp_dists)

                # 使用最新的投影函数对新数据进行投影
                new_embeddings = self._model.reuse_project(cdist(data, total_cntp_points))
                total_embeddings = np.concatenate([prev_updated_embeddings, new_embeddings], axis=0)

                # 更新后的之前所有control points的投影结果，需要与之前的投影结果对齐
                updated_pre_cntp_embeddings = total_embeddings[self._pre_control_indices]
                # 使用Procrustes analysis保持mental-map
                aligned_total_embeddings = procrustes_analysis(self._pre_cntp_embeddings, updated_pre_cntp_embeddings,
                                                               total_embeddings, align=True)

                self._pre_control_indices = np.concatenate([self._pre_control_indices, cntp_indices + pre_data_num])
                self._pre_control_points = total_cntp_points
                self._pre_cntp_embeddings = aligned_total_embeddings[self._pre_control_indices]
                self._pre_embedding = aligned_total_embeddings

            self._model_return_queue.put([copy(self._model), self._pre_embedding, self._pre_control_indices,
                                          self._pre_control_points])
