import warnings

import h5py
import numpy as np
import skdim
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.graph import _fix_connected_components

from model.incrementalLE import kNNBasedIncrementalMethods
from model.scdr.dependencies.experiment import position_vis
from utils.nn_utils import compute_knn_graph


class SIsomap(kNNBasedIncrementalMethods, Isomap):
    def __init__(self, train_num, n_components, n_neighbors):
        Isomap.__init__(self, n_neighbors=n_neighbors, n_components=n_components)
        kNNBasedIncrementalMethods.__init__(self, train_num, n_components, n_neighbors, single=True)
        self.G = None

    def _first_train(self, train_data):
        self.pre_embeddings = self.fit_transform(train_data)
        self.G = self.dist_matrix_ ** 2
        self.G *= -0.5
        return self.pre_embeddings

    def _incremental_embedding(self, new_data):
        pre_data_num = self.G.shape[0]
        new_data = new_data[np.newaxis, :]
        knn_indices, knn_dists, dists = self._cal_new_data_kNN(new_data, include_self=False)
        geodesic_dists = self._cal_new_data_geodesic_distance(pre_data_num, knn_indices.squeeze(), knn_dists.squeeze())
        embedding = self._embedding_new_data(pre_data_num, geodesic_dists)
        self.pre_embeddings = np.concatenate([self.pre_embeddings, embedding], axis=0)
        geodesic_dists_self = np.append(geodesic_dists, 0.0)
        self.G = np.concatenate([self.G, geodesic_dists[:, np.newaxis]], axis=1)
        self.G = np.concatenate([self.G, geodesic_dists_self[np.newaxis, :]], axis=0)

        return self.pre_embeddings

    def _cal_new_data_geodesic_distance(self, pre_data_num, knn_indices, knn_dists):
        knn_dists **= 2
        geodesic_dists = np.zeros(pre_data_num)
        for i in range(pre_data_num):
            min_dist = knn_dists[0] + self.G[knn_indices[0]][i]
            for j in range(1, self.n_neighbors):
                min_dist = min(min_dist, knn_dists[j] + self.G[knn_indices[j]][i])
            geodesic_dists[i] = min_dist
        return geodesic_dists

    def _embedding_new_data(self, pre_data_num, new_data_g):
        one_vector = np.ones(pre_data_num)
        # n * 1
        c = 0.5 * (np.mean(new_data_g) * one_vector - new_data_g - np.mean(self.G) * one_vector + np.mean(self.G, axis=1))
        # 2 * 1
        p = np.dot(np.dot(np.linalg.inv(np.dot(self.pre_embeddings.T, self.pre_embeddings)),
                          self.pre_embeddings.T), c[:, np.newaxis])
        y_star = np.concatenate([self.pre_embeddings, p.T], axis=0)
        embedding = p - np.mean(y_star, axis=0)[:, np.newaxis]
        return embedding.T


class SIsomapPlus(kNNBasedIncrementalMethods):
    def __init__(self, train_num, n_components, n_neighbors):
        kNNBasedIncrementalMethods.__init__(self, train_num, n_components, n_neighbors)


if __name__ == '__main__':
    with h5py.File("../../../Data/H5 Data/food.h5", "r") as hf:
        X = np.array(hf['x'])
        Y = np.array(hf['y'])

    train_num = 500
    train_data = X[:train_num]
    train_labels = Y[:train_num]

    ile = SIsomap(train_num, 2, 10)

    first_embeddings = ile.fit_new_data(train_data)
    position_vis(train_labels, None, first_embeddings, "first")

    second_embeddings = ile.fit_new_data(X[train_num:train_num + 500])
    position_vis(Y[train_num:train_num + 500], None, second_embeddings[train_num:], "second new")
    position_vis(train_labels, None, second_embeddings[:train_num], "second pre")
    position_vis(Y[:train_num + 500], None, second_embeddings, "second whole")