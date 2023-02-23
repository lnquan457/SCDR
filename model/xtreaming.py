import time

import numpy as np
from procrustes import orthogonal
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

from model.dr_models.upd import cal_dist, UPDis4Streaming


def sampled_control_points(data):
    n_samples = data.shape[0]
    km = KMeans(n_clusters=int(np.sqrt(n_samples)))
    km.fit(data)
    centroids = km.cluster_centers_
    # len(centroids) * n
    # dists = np.linalg.norm(np.repeat(np.expand_dims(data, axis=0), axis=0, repeats=centroids.shape[0]) - \
    #                        np.repeat(np.expand_dims(centroids, axis=1), axis=1, repeats=n_samples), axis=-1)
    dists = cdist(centroids, data)

    sampled_indices = np.argsort(dists, axis=-1)[:, 0]
    sampled_x = data[sampled_indices]
    return sampled_x, sampled_indices


def procrustes_analysis(pre_data, cur_data, all_data, align=True, scale=True, translate=True):
    if not align:
        return all_data
    # 在之前control points和当前更新后的control points之间进行普氏分析，得到正交矩阵Q
    # 使得第一个参数尽量与第二个参数对齐
    result = orthogonal(cur_data, pre_data, scale=scale, translate=translate)
    # print("Procrustes Error = ", result.error)
    # t是一个2 * 2的正交矩阵，new_a是经过放缩和平移之后的
    aligned_embeddings = np.dot(all_data, result.t)

    return aligned_embeddings


class XtreamingModel:
    def __init__(self, buffer_size=100, eta=0.99, window_size=2000):
        self.buffer_size = buffer_size
        self._eta = eta
        self.pro_model = UPDis4Streaming(eta)
        self.buffered_data = None
        self.time_step = 0
        self.pre_control_indices = None
        self.pre_control_points = None
        self.pre_embedding = None
        self.pre_cntp_embeddings = None
        self.lof = LocalOutlierFactor(n_neighbors=10, novelty=True, metric="euclidean", contamination=0.1)
        self.time_costs = 0
        self._time_cost_records = [0]
        self._window_size = window_size
        self._total_n_samples = 0
        self._total_out_num = 0
        self._drift_history = []

        self.total_data = None

    def _buffering(self, data):
        if self.buffered_data is None:
            self.buffered_data = data
        else:
            self.buffered_data = np.concatenate([self.buffered_data, data], axis=0)

        self.time_step += 1
        return self.buffered_data.shape[0] >= self.buffer_size

    def fit_new_data(self, data, labels=None):
        key_time = 0
        sta = time.time()
        if not self._buffering(data):
            return self.pre_embedding, 0, False, 0
        key_time += time.time() - sta
        self.total_data = self.buffered_data if self.total_data is None else np.concatenate([self.total_data,
                                                                                             self.buffered_data], axis=0)
        self._total_n_samples += self.buffered_data.shape[0]

        sta = time.time()
        ret = self.fit()
        out_num = self._slide_window()
        key_time += time.time() - sta
        return ret, key_time, True, out_num

    def _slide_window(self):
        out_num = max(0, self._total_n_samples - self._window_size)
        if out_num > 0:
            self._total_out_num += out_num
            self._total_n_samples -= out_num
            self.pre_embedding = self.pre_embedding[out_num:]
            self.total_data = self.total_data[out_num:]
        return out_num

    def model_slide(self, new_embeddings, out_num):
        self.pre_control_indices -= out_num
        valid_indices = np.argwhere(self.pre_control_indices >= 0).squeeze()
        self.pre_control_indices = self.pre_control_indices[valid_indices].squeeze()
        self.pre_control_points = self.pre_control_points[valid_indices].squeeze()
        self.pre_cntp_embeddings = new_embeddings[self.pre_control_indices].squeeze()

        self.pro_model.control_points = self.pro_model.control_points[valid_indices]
        self.pro_model.ctp2ctp_dists = self.pro_model.ctp2ctp_dists[valid_indices][:, valid_indices]
        self.pro_model.control_embeddings = self.pro_model.control_embeddings[valid_indices]
        self.pro_model.normed_eigen_vector = self.pro_model.normed_eigen_vector[:, valid_indices]

    def buffer_empty(self):
        return self.buffered_data is None

    def fit(self):
        medoids, sampled_indices = sampled_control_points(self.buffered_data)
        if self.pre_embedding is None:
            self._initial_project(medoids, sampled_indices)
        else:
            """
            首先需要采样，然后从采样的代表点中以此判断是否分布发生变化，对于分布发生变化的点，应该将其加入到代表点集合中
            但是这里需要注意，代表点集合是在时刻变化的
            
            需要维持一个专门给lof用的代表点集合，然后动态更新这个代表点集合。如果集合发生改变则需要重新拟合LoF
            然后再对当前buffer的数据进行预测，
            """
            cur_control_points, drift_indices = self._detect_concept_drift(medoids)
            cur_control_indices = sampled_indices[drift_indices]
            dists2cntp = cdist(self.buffered_data, self.pre_control_points)

            if cur_control_points is None:
                # 使用之前的映射函数进行投影
                cur_embeddings = self.pro_model.reuse_project(dists2cntp)
                self.pre_embedding = np.concatenate([self.pre_embedding, cur_embeddings], axis=0)
            else:
                print("Re Projection !")
                # valid_history_data = self.total_data[-self._window_size:]
                # valid_centroids, valid_sampled_indices = sampled_control_points(valid_history_data)
                # new_dists2cntp = cdist(valid_history_data, self.pre_control_points)
                # new_control_indices = sampled_indices[drift_indices]

                aligned_total_embeddings, total_cntp_points = self._re_projection(dists2cntp, cur_control_indices,
                                                                                  cur_control_points)
                # aligned_total_embeddings, total_cntp_points = self._re_projection_slide()

                self.pre_control_points = total_cntp_points
                self.pre_embedding = aligned_total_embeddings
                # print(self.pre_control_indices)
                # self.model_slide(aligned_total_embeddings, self._total_out_num)
                # print(self.pre_control_points.shape)
                self.lof.fit(self.pre_control_points)
                self._total_out_num = 0

        self.buffered_data = None

        return self.pre_embedding

    def _initial_project(self, medoids, sampled_indices):
        # dists = cal_dist(self.buffered_data)
        dists = cdist(self.buffered_data, self.buffered_data)
        cntp2cntp_dists = dists[sampled_indices, :][:, sampled_indices]
        dists2cntp = dists[:, sampled_indices]
        self.pre_control_points = medoids
        self.pre_control_indices = sampled_indices
        self.pre_embedding = self.pro_model.fit_transform(dists2cntp, cntp2cntp_dists)
        # self.pre_embedding = self.pro_model.fit_transform_cntp_free(dists)
        self.pre_cntp_embeddings = self.pre_embedding[sampled_indices]
        self.buffered_data = None
        self.lof.fit(self.pre_control_points)

    def _detect_concept_drift(self, cur_medoids):
        # self.lof.fit(self.pre_control_points)
        labels = self.lof.predict(cur_medoids)
        # print(labels)
        control_indices = np.where(labels == -1)[0]
        cur_control_points = None if len(control_indices) == 0 else cur_medoids[control_indices.astype(int)]
        return cur_control_points, control_indices

    def _re_projection(self, dists2cntp, control_indices, control_points):
        pre_data = self.total_data[:self.pre_embedding.shape[0]]
        new_pre_cntp_points, new_pre_cntp_indices = sampled_control_points(pre_data)
        new_pre_cntp_embeddings = self.pre_embedding[new_pre_cntp_indices]
        # cur_cntp_embeddings = self.pro_model.reuse_project(dists2cntp[control_indices])
        # total_cntp_embeddings = np.concatenate([new_pre_cntp_embeddings, cur_cntp_embeddings], axis=0)
        total_cntp_points = np.concatenate([new_pre_cntp_points, control_points], axis=0)
        pre_dists2all_cntp = cdist(pre_data, total_cntp_points)
        cntp2cntp_dists = cdist(total_cntp_points, total_cntp_points)
        updated_pre_embeddings = self.pro_model.fit_transform(pre_dists2all_cntp, cntp2cntp_dists)
        new_data_embeddings = self.pro_model.reuse_project(cdist(self.buffered_data, total_cntp_points))
        updated_total_embeddings = np.concatenate([updated_pre_embeddings, new_data_embeddings], axis=0)
        updated_pre_cntp_embeddings = updated_total_embeddings[new_pre_cntp_indices]

        aligned_total_embeddings = procrustes_analysis(new_pre_cntp_embeddings,
                                                       updated_pre_cntp_embeddings, updated_total_embeddings, align=True)

        self.pre_control_indices = np.concatenate([new_pre_cntp_indices, control_indices + pre_data.shape[0]])
        return aligned_total_embeddings, total_cntp_points

    def _re_projection_no_slide(self, dists2cntp, control_indices, control_points):

        pre_data_num = self.pre_embedding.shape[0]
        total_cntp_points = np.concatenate([self.pre_control_points, control_points], axis=0)

        # 求出之前的所有数据到之前的控制点的距离
        dists2pre_cntp = cdist(self.pre_embedding, self.pre_cntp_embeddings)
        # 求出之前的所有数据到当前控制点的距离
        cur_cntp_embeddings = self.pro_model.reuse_project(dists2cntp[control_indices])
        dists2new_cntp = cdist(self.pre_embedding, cur_cntp_embeddings)
        pre_dists2all_cntp = np.concatenate([dists2pre_cntp, dists2new_cntp], axis=1)

        # 所有control points之间的距离，用于构造投影函数
        total_cntp_embeddings = np.concatenate([self.pre_cntp_embeddings, cur_cntp_embeddings], axis=0)
        cntp_dists = cal_dist(total_cntp_embeddings)
        # 对之前的数据以及所有control points进行重新投影
        updated_pre_embeddings = self.pro_model.fit_transform(pre_dists2all_cntp, cntp_dists)

        # 使用最新的投影函数对新数据进行投影
        new_embeddings = self.pro_model.reuse_project(cdist(self.buffered_data, total_cntp_points))
        total_embeddings = np.concatenate([updated_pre_embeddings, new_embeddings], axis=0)

        # 更新后的之前所有control points的投影结果，需要与之前的投影结果对齐
        # updated_pre_cntp_embeddings = total_embeddings[self.pre_control_indices]
        updated_pre_cntp_embeddings = self.pro_model.reuse_project(cdist(self.pre_control_points, total_cntp_points))
        # 使用Procrustes analysis保持mental-map
        aligned_total_embeddings = procrustes_analysis(self.pre_cntp_embeddings,
                                                       updated_pre_cntp_embeddings, total_embeddings, align=True)

        self.pre_control_indices = np.concatenate([self.pre_control_indices, control_indices + pre_data_num])
        return aligned_total_embeddings, total_cntp_points

    def transform(self, data):
        dists = cdist(data, self.pre_control_points)
        embeddings = self.pro_model.reuse_project(dists)
        return embeddings

    def ending(self):
        output = "Time Cost: %.4f" % self.time_costs
        print(output)
        return output, self._time_cost_records
