import numpy as np
from scipy.spatial.distance import cdist
from sklearn.manifold import SpectralEmbedding

from dataset.warppers import arr_move_one
from model.scdr.dependencies.scdr_utils import StreamingDataRepo


class IncrementalLE(SpectralEmbedding):
    def __init__(self, train_num, n_components, n_neighbors):
        SpectralEmbedding.__init__(self, n_components, n_neighbors=n_neighbors)
        self.train_num = train_num
        self.knn_indices = None
        self.knn_dists = None
        self.weight_matrix = None
        self.pre_embeddings = None
        self.stream_dataset = StreamingDataRepo(n_neighbors)
        self.trained = False

    def first_train(self, train_data, train_labels=None):
        le_embedder = SpectralEmbedding(self.n_components, n_neighbors=self.n_neighbors)
        self.pre_embeddings = le_embedder.fit_transform(train_data)
        affinity = le_embedder.affinity
        self.weight_matrix = le_embedder.affinity_matrix_
        print(affinity)
        return self.pre_embeddings

    def fit_new_data(self, x, labels=None):
        self.stream_dataset.add_new_data(x, None, labels)

        if not self.trained:
            if self.stream_dataset.get_n_samples() < self.train_num:
                return None
            self.trained = True
            return self.first_train(self.stream_dataset.total_data, self.stream_dataset.total_label)

        new_data_embeddings = self._incremental_embedding(x)
        total_embeddings = np.concatenate([self.pre_embeddings, new_data_embeddings], axis=0)
        return total_embeddings

    def _incremental_embedding(self, new_data):
        self._update_kNN(new_data)
        self._update_weight_matrix()
        return self._embedding_new_data_linear()

    def _update_kNN(self, new_data):
        # 1. 计算新数据的kNN
        knn_indices, knn_dists, dists = self._cal_new_data_kNN(new_data)


    def _update_weight_matrix(self):
        pass

    def _embedding_new_data_linear(self):
        pass

    def _embedding_new_data_manifold(self):
        pass

    def _cal_new_data_kNN(self, new_data):
        new_data_num = new_data.shape[0]
        pre_data = self.stream_dataset.total_data[:-new_data_num]
        dists = cdist(new_data, pre_data)
        knn_indices = np.argsort(dists, axis=1)[:, 1:self.n_neighbors + 1]
        knn_dists = np.zeros(shape=(new_data_num, self.n_neighbors))
        for i in range(new_data_num):
            knn_dists[i] = dists[i][knn_indices[i]]
        return knn_indices, knn_dists, dists

    def _update_old_data_kNN(self, new_data_num, dists, new_knn_indices, new_knn_dists):
        farest_neighbor_dist = self.knn_dists[:, -1]
        pre_n_samples = self.stream_dataset.get_n_samples() - new_data_num
        neighbor_changed_indices = []
        for i in range(new_data_num):
            indices = np.where(dists[i] < farest_neighbor_dist)[0]
            if len(indices) <= 0:
                continue

            for j in indices:
                if j == pre_n_samples + i:
                    continue
                if j not in neighbor_changed_indices:
                    neighbor_changed_indices.append(j)
                # 为当前元素找到一个插入位置即可，即distances中第一个小于等于dists[i][j]的元素位置，始终保持distances有序，那么最大的也就是最后一个
                insert_index = self.n_neighbors - 1
                while insert_index >= 0 and dists[i][j] <= self.knn_dists[j][insert_index]:
                    insert_index -= 1

                if self.knn_indices[j][-1] not in neighbor_changed_indices:
                    neighbor_changed_indices.append(self.knn_indices[j][-1])

                # 这个更新的过程应该是迭代的，distance必须是递增的, 将[insert_index+1: -1]的元素向后移一位
                arr_move_one(self.knn_dists[j], insert_index + 1, dists[i][j])
                arr_move_one(self.knn_indices[j], insert_index + 1, pre_n_samples + i)
                self.farest_neighbor_dist[j] = self.knn_distances[j][-1]