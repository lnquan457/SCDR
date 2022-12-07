import time

import h5py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold._locally_linear import barycenter_weights
from sklearn.neighbors import kneighbors_graph

from dataset.warppers import arr_move_one, KNNManager, extract_csr, DataRepo
from model.scdr.dependencies.experiment import position_vis
from utils.nn_utils import compute_knn_graph


class kNNBasedIncrementalMethods:
    def __init__(self, train_num, n_components, n_neighbors, single=False):
        self.single = single
        self.initial_train_num = train_num
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.knn_manager = KNNManager(n_neighbors)
        self.stream_dataset = DataRepo(n_neighbors)
        self.pre_embeddings = None
        self.trained = False
        self.time_cost = 0
        self._time_cost_records = [0]

    def _update_kNN(self, new_data):
        # 1. 计算新数据的kNN
        knn_indices, knn_dists, dists = self._cal_new_data_kNN(new_data)
        self.knn_manager.add_new_kNN(knn_indices, knn_dists)

        # 2. 更新旧数据的kNN
        new_data_num = new_data.shape[0]
        pre_data_num = self.stream_dataset.get_n_samples() - new_data_num
        neighbor_changed_indices = self.knn_manager.update_previous_kNN(new_data_num, pre_data_num, dists,
                                                                        data_num_list=[0, new_data_num])
        neighbor_changed_indices = np.array(neighbor_changed_indices)
        return knn_indices, knn_dists, neighbor_changed_indices[neighbor_changed_indices < pre_data_num]

    def _cal_new_data_kNN(self, new_data, include_self=True):
        new_data_num = new_data.shape[0]
        dists = cdist(new_data, self.stream_dataset.get_total_data())
        if include_self:
            knn_indices = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        else:
            knn_indices = np.argsort(dists, axis=1)[:, 1:self.n_neighbors + 1]
        knn_dists = np.zeros(shape=(new_data_num, self.n_neighbors))
        for i in range(new_data_num):
            knn_dists[i] = dists[i][knn_indices[i]]
        return knn_indices, knn_dists, dists

    def _first_train(self, train_data):
        pass

    def fit_new_data(self, x, labels=None):
        if self.single:
            return self._fit_new_data_single(x, labels)
        else:
            return self._fit_new_data_batch(x, labels)

    def _fit_new_data_batch(self, x, labels=None):
        self.stream_dataset.add_new_data(x, None, labels)

        if not self.trained:
            if self.stream_dataset.get_n_samples() < self.initial_train_num:
                return None
            self.trained = True
            self._first_train(self.stream_dataset.get_total_data())
        else:
            sta = time.time()
            self._incremental_embedding(x)
            self.time_cost += time.time() - sta
            self._time_cost_records.append(time.time() - sta + self._time_cost_records[-1])

        return self.pre_embeddings

    def _fit_new_data_single(self, x, labels=None):
        if not self.trained:
            self.stream_dataset.add_new_data(x, None, labels)
            if self.stream_dataset.get_n_samples() >= self.initial_train_num:
                self.trained = True
                self._first_train(self.stream_dataset.get_total_data())
        else:
            for i, item in enumerate(x):
                self.stream_dataset.add_new_data(np.reshape(item, (1, -1)), None, labels[i] if labels is not None else None)
                sta = time.time()
                self._incremental_embedding(item)
                self.time_cost += time.time() - sta

        return self.pre_embeddings

    def _incremental_embedding(self, new_data):
        pass

    def ending(self):
        output = "Time Cost: %.4f" % self.time_cost
        print(output)
        return output, self._time_cost_records


class IncrementalLE(SpectralEmbedding, kNNBasedIncrementalMethods):
    def __init__(self, train_num, n_components, n_neighbors):
        SpectralEmbedding.__init__(self, n_components, n_neighbors=n_neighbors)
        kNNBasedIncrementalMethods.__init__(self, train_num, n_components, n_neighbors)
        # 只在训练时用到
        self.initial_knn_indices = None
        self.initial_knn_dists = None
        self.weight_matrix = None

    def _first_train(self, train_data):
        self.pre_embeddings = self.fit_transform(train_data)
        self.knn_manager.add_new_kNN(self.initial_knn_indices, self.initial_knn_dists)
        self.weight_matrix = np.array(self.affinity_matrix_.todense())
        return self.pre_embeddings

    def _incremental_embedding(self, new_data):
        pre_knn_indices = self.knn_manager.knn_indices
        new_knn_indices, new_knn_dists, neighbor_changed_indices = self._update_kNN(new_data)

        self._update_weight_matrix(pre_knn_indices, new_knn_indices, neighbor_changed_indices)

        self.pre_embeddings = self._embedding_new_data_linear(new_data.shape[0])
        self._update_previous_embeddings(neighbor_changed_indices)
        return self.pre_embeddings

    # 仅体现0-1连接关系
    def _update_weight_matrix(self, pre_knn_indices, new_knn_indices, neighbor_changed_indices):
        new_data_num = new_knn_indices.shape[0]
        pre_data_num = pre_knn_indices.shape[0]
        self.weight_matrix = np.concatenate([self.weight_matrix, np.zeros((self.stream_dataset.get_n_samples() -
                                                                           new_data_num, new_data_num))], axis=1)
        new_weight_matrix = np.zeros(shape=(new_data_num, self.stream_dataset.get_n_samples()))
        self.weight_matrix = np.concatenate([self.weight_matrix, new_weight_matrix], axis=0)

        for i in neighbor_changed_indices:
            diff = np.setdiff1d(pre_knn_indices[i], self.knn_manager.knn_indices[i])
            new = np.setdiff1d(self.knn_manager.knn_indices[i], pre_knn_indices[i])
            for item in diff:
                if self.weight_matrix[i][item] == 1:  # 说明i和item互为邻居
                    self.weight_matrix[i][item] = 0.5
                    self.weight_matrix[item][i] = 0.5
                else:   # 等于0.5说明item的kNN中没有i
                    self.weight_matrix[i][item] = 0
                    self.weight_matrix[item][i] = 0

            # 均是新到达的数据，这些新数据之前不存在连接关系
            for item in new:
                self.weight_matrix[i][item] = 0.5
                self.weight_matrix[item][i] = 0.5

        for i in range(new_data_num):
            zero_indices = np.argwhere(self.weight_matrix[new_knn_indices[i], i+pre_data_num] == 0)
            half_indices = np.argwhere(self.weight_matrix[new_knn_indices[i], i+pre_data_num] == 0.5)
            self.weight_matrix[new_knn_indices[i][zero_indices], i+pre_data_num] = 0.5
            self.weight_matrix[i+pre_data_num, new_knn_indices[i][zero_indices]] = 0.5
            self.weight_matrix[new_knn_indices[i][half_indices], i + pre_data_num] = 1
            self.weight_matrix[i + pre_data_num, new_knn_indices[i][half_indices]] = 1
            self.weight_matrix[i+pre_data_num, i+pre_data_num] = 1

    def _embedding_new_data_linear(self, new_data_num):
        new_embeddings = np.zeros((new_data_num, self.n_components))
        embeddings = np.concatenate([self.pre_embeddings, new_embeddings], axis=0)

        for i in range(new_data_num):
            index = -new_data_num + i
            embeddings[index] = \
                np.dot(embeddings[:index].T, np.expand_dims(self.weight_matrix[index, :index], axis=1)).T \
                / np.sum(self.weight_matrix[index, :index])

        return embeddings

    def _embedding_new_data_manifold(self):
        pass

    def _update_previous_embeddings(self, neighbor_changed_indices):
        neighbor_changed_data = self.stream_dataset._total_data[neighbor_changed_indices]
        indices = self.knn_manager.knn_indices[neighbor_changed_indices]
        weights = barycenter_weights(neighbor_changed_data, self.stream_dataset._total_data, indices)

        for i, idx in enumerate(neighbor_changed_indices):
            self.pre_embeddings[idx] = np.sum(np.expand_dims(weights[i], axis=1) * self.pre_embeddings[indices[i]], axis=0)

    def _get_affinity_matrix(self, X, Y=None):
        self.n_neighbors_ = (
            self.n_neighbors
            if self.n_neighbors is not None
            else max(int(X.shape[0] / 10), 1)
        )
        self.affinity_matrix_ = kneighbors_graph(
            X, self.n_neighbors_, include_self=True, n_jobs=self.n_jobs, mode="distance", metric="euclidean"
        )
        knn_indices, knn_dists = extract_csr(self.affinity_matrix_, np.arange(X.shape[0]), norm=False)
        self.initial_knn_indices = np.array(knn_indices, dtype=int)
        self.initial_knn_dists = np.array(knn_dists, dtype=float)

        self.affinity_matrix_.data = self.affinity_matrix_.data * 0 + 1

        self.affinity_matrix_ = 0.5 * (
            self.affinity_matrix_ + self.affinity_matrix_.T
        )
        return self.affinity_matrix_


if __name__ == '__main__':
    with h5py.File("../../../Data/H5 Data/food.h5", "r") as hf:
        X = np.array(hf['x'])
        Y = np.array(hf['y'])

    train_num = 2000
    train_data = X[:train_num]
    train_labels = Y[:train_num]

    ile = IncrementalLE(train_num, 2, 10)

    first_embeddings = ile.fit_new_data(train_data)
    position_vis(train_labels, None, first_embeddings, "first")

    second_embeddings = ile.fit_new_data(X[train_num:train_num + 1000])
    position_vis(Y[train_num:train_num + 1000], None, second_embeddings[train_num:], "second - new")
    position_vis(train_labels, None, second_embeddings[:train_num], "second - pre")
