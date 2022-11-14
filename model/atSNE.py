#!/bin/bash
from __future__ import division  # Python 2 users only

__doc__ = """ Script for implementation of approximated tSNE algorithm.
See "Approximated and User Steerable tSNE for Progressive Visual Analytics" """

import queue
import warnings
from time import time
from annoy import AnnoyIndex
import h5py
import scipy as sp
import numpy as np
from scipy.sparse import isspmatrix, csr_matrix
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold._t_sne import _joint_probabilities_nn, _joint_probabilities, MACHINE_EPSILON, _gradient_descent, \
    _kl_divergence_bh
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree, KDTree
from sklearn.manifold import TSNE, _utils
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import check_array, check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads


def draw_projections(z, c, vis_save_path=None):
    x = z[:, 0]
    y = z[:, 1]
    c = np.array(c, dtype=int)

    plt.figure(figsize=(8, 8))
    if c is None:
        sns.scatterplot(x=x, y=y, s=1, legend=False, alpha=0.9)
    else:
        classes = np.unique(c)
        num_classes = classes.shape[0]
        palette = "tab10" if num_classes <= 10 else "tab20"
        sns.scatterplot(x=x, y=y, hue=c, s=8, palette=palette, legend=False, alpha=1.0)

    plt.xticks([])
    plt.yticks([])
    if vis_save_path is not None:
        plt.savefig(vis_save_path, dpi=800, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def my_joint_probabilities_nn(distances, desired_perplexity, verbose):
    t0 = time()
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances_data, desired_perplexity, verbose
    )
    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"
    P = csr_matrix((conditional_P.ravel(), distances.indices, distances.indptr), shape=(n_samples, n_samples))
    P = P + P.T
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        duration = time() - t0
        print("[t-SNE] Computed conditional probabilities in {:.3f}s".format(duration))
    return P, conditional_P


class atSNEModel(TSNE):

    def __init__(self, finetune_iter, *args, **kwargs):
        self.finetune_iter = finetune_iter
        super(atSNEModel, self).__init__(*args, **kwargs)
        self.n_iter_ = 0
        self.kl_divergence_ = 0
        self.degrees_of_freedom = 1

        self.first_fitted = False
        self.X = None
        self.distances_nn = None
        self.neighbors_nn = None
        self.max_dist = None
        self.pre_embeddings = None

    def get_all_knn(self, k):
        cur_knn_indices = []
        cur_knn_dists = []
        for i in range(self.annoy_index.get_n_items()):
            indices, dists = self.annoy_index.get_nns_by_item(i, k, include_distances=True)
            cur_knn_indices.append(indices)
            cur_knn_dists.append(dists)
        return np.array(cur_knn_indices, dtype=int), np.array(cur_knn_dists, dtype=float)

    def get_knn_by_items(self, items, k):
        cur_knn_indices = []
        cur_knn_dists = []
        for data in items:
            indices, dists = self.annoy_index.get_nns_by_vector(data, k, include_distances=True)
            cur_knn_indices.append(indices)
            cur_knn_dists.append(dists)
        return np.array(cur_knn_indices, dtype=int), np.array(cur_knn_dists, dtype=float)

    def add_items2annoy(self, items):
        pre_num = self.annoy_index.get_n_items()
        self.annoy_index.unbuild()
        for j in range(items.shape[0]):
            self.annoy_index.add_item(j+pre_num, items[j])
        self.annoy_index.build(100)

    def _fit(self, X, skip_num_points=0):
        random_state = check_random_state(self.random_state)

        if self.metric == "precomputed":
            if isinstance(self.init, str) and self.init == 'pca':
                raise ValueError("The parameter init=\"pca\" cannot be used "
                                 "with metric=\"precomputed\".")
            distances = X
        else:
            if self.verbose:
                print("[t-SNE] Computing pairwise distances...")

            if self.metric == "euclidean":
                distances = pairwise_distances(X, metric=self.metric,
                                               squared=True)
            else:
                distances = pairwise_distances(X, metric=self.metric)

        if self.learning_rate == "warn":
            self._learning_rate = 200.0
        else:
            self._learning_rate = self.learning_rate

        self.n_samples, dim = X.shape
        # the number of nearest neighbors to find
        k = min(self.n_samples - 1, int(3. * self.perplexity + 1))
        self.k = k
        self.X = X
        neighbors_nn = None
        if self.method == 'barnes_hut':
            if self.verbose:
                print("[t-SNE] Computing %i nearest neighbors..." % k)
            if self.metric == 'precomputed':
                neighbors_nn = np.argsort(distances, axis=1)[:, :k]
                distances_nn = distances[:, :k]
            else:
                self.annoy_index = AnnoyIndex(dim, 'euclidean')
                for j in range(self.n_samples):
                    self.annoy_index.add_item(j, X[j])
                self.annoy_index.build(100)
                neighbors_nn, distances_nn = self.get_all_knn(self.k)

                # self.myflann = FLANN()
                # self.flann_params = self.myflann.build_index(X, algorithm="autotuned", target_precision=self.rho,
                #                                              log_level='info')
                # neighbors_nn, distances_nn = self.myflann.nn_index(X, self.k + 1, checks=self.flann_params["checks"])

                neighbors_nn = neighbors_nn[:, 1:]
                distances_nn = distances_nn[:, 1:]

            self.neighbors_nn = neighbors_nn
            self.distances_nn = distances_nn
            self.max_dist = np.max(distances_nn, axis=1)
            coo_row = np.expand_dims(np.arange(0, self.n_samples, 1), axis=1).repeat(self.k - 1, axis=1).ravel()
            csr_distances = sp.sparse.coo_matrix((np.ravel(distances_nn), (coo_row, np.ravel(neighbors_nn))),
                                                 shape=(self.n_samples, self.n_samples)).tocsr()
            self.P = _joint_probabilities_nn(csr_distances, self.perplexity, self.verbose)
        else:
            self.P = _joint_probabilities(distances, self.perplexity, self.verbose)
            assert np.all(np.isfinite(self.P)), "All probabilities should be finite"
            assert np.all(self.P >= 0), "All probabilities should be zero or positive"
            assert np.all(self.P <= 1), ("All probabilities should be less "
                                         "or then equal to one")

        if isinstance(self.init, np.ndarray):
            x_embedded = self.init
        elif self.init == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            x_embedded = pca.fit_transform(X)
        elif self.init == 'random':
            x_embedded = None
        else:
            raise ValueError("Unsupported initialization scheme: %s" % self.init)

        self.pre_embeddings = self._tsne(self.P, self.degrees_of_freedom, self.n_samples,
                                         X_embedded=x_embedded,
                                         neighbors=neighbors_nn,
                                         skip_num_points=skip_num_points)
        return self.pre_embeddings

    def fit_new_data(self, data, labels=None, ex_ratio=0.1):
        if not self.first_fitted:
            self.first_fitted = True
            return self.fit_transform(data, labels)

        total_data = np.concatenate([self.X, data], axis=0)
        new_total_n_samples = total_data.shape[0]

        # new_flann = FLANN()
        # new_flann.build_index(total_data, algorithm="autotuned", target_precision=self.rho,
        #                       log_level='info')
        # neighbors_nn, distances = new_flann.nn_index(data, self.k, checks=self.flann_params["checks"])

        # 只需要获得新的数据的k近邻即可
        # neighbors_nn, distances = self.myflann.nn_index(data, self.k, checks=self.flann_params["checks"])

        neighbors_nn, distances = self.get_knn_by_items(data, self.k)
        distances **= 2

        neighbors_nn = neighbors_nn[:, 1:]
        distances_nn = distances[:, 1:]

        cur_p, cur_cond_p, new_distance_nn, new_neighbor_nn = self.update_current_knn(data, distances_nn,
                                                                                      neighbors_nn,
                                                                                      new_total_n_samples)

        new_x_initial_embeddings = self.initialize_embeddings(cur_cond_p, neighbors_nn)

        new_embeddings = self._finetune_tsne(cur_p, ex_ratio, self.degrees_of_freedom,
                                             new_total_n_samples, new_x_initial_embeddings)

        self.update_related_data(new_distance_nn, new_embeddings, new_neighbor_nn, total_data)

        sta = time()
        # self.myflann.build_index(total_data, algorithm="autotuned", target_precision=self.rho, log_level='none')
        self.add_items2annoy(data)
        # print("建树耗时：", time() - sta)
        return new_embeddings

    def update_related_data(self, new_distance_nn, new_embeddings, new_neighbor_nn, total_data):
        self.X = total_data
        self.distances_nn = new_distance_nn
        self.neighbors_nn = new_neighbor_nn
        self.max_dist = np.max(self.distances_nn, axis=1)
        self.pre_embeddings = new_embeddings

    def initialize_embeddings(self, cur_cond_p, neighbors_nn):
        new_n_samples = neighbors_nn.shape[0]
        pre_n_samples = self.X.shape[0]
        new_data_initial_embeddings = np.empty((new_n_samples, self.n_components))
        for i in range(new_n_samples):
            new_data_initial_embeddings[i] = np.sum(self.pre_embeddings[neighbors_nn[i]] *
                                                    np.expand_dims(cur_cond_p[pre_n_samples + i], axis=1).
                                                    repeat(self.n_components, axis=1), axis=0)
        new_x_initial_embeddings = np.concatenate([self.pre_embeddings, new_data_initial_embeddings], axis=0)
        return new_x_initial_embeddings

    def update_current_knn(self, data, distances_nn, neighbors_nn, new_total_n_samples):
        pre_n_samples = self.X.shape[0]
        new_n_samples = data.shape[0]
        # dists = np.linalg.norm(np.expand_dims(data, axis=1).repeat(repeats=pre_n_samples, axis=1) -
        #                        np.expand_dims(self.X, axis=0).repeat(repeats=new_n_samples, axis=0), axis=-1)
        dists = cdist(data, self.X)
        dists **= 2
        for i in range(new_n_samples):
            indices = np.where(dists[i] - self.max_dist < 0)[0]
            if len(indices) <= 0:
                continue
            for j in indices:
                t = np.argmax(self.distances_nn[j])
                self.distances_nn[j][t] = dists[i][j]
                self.neighbors_nn[j][t] = pre_n_samples + i
                re_indices = np.argsort(self.distances_nn[j])
                self.distances_nn[j] = self.distances_nn[j][re_indices]
                self.neighbors_nn[j] = self.neighbors_nn[j][re_indices]
                self.max_dist[j] = dists[i][j]
        new_distance_nn = np.concatenate([self.distances_nn, distances_nn], axis=0)
        new_neighbor_nn = np.concatenate([self.neighbors_nn, neighbors_nn], axis=0)
        coo_row = np.expand_dims(np.arange(0, new_total_n_samples, 1), axis=1).repeat(self.k - 1, axis=1).ravel()
        csr_distances = sp.sparse.coo_matrix((np.ravel(new_distance_nn), (coo_row, np.ravel(new_neighbor_nn))),
                                             shape=(new_total_n_samples, new_total_n_samples)).tocsr()
        cur_P, cur_cond_P = my_joint_probabilities_nn(csr_distances, self.perplexity, self.verbose)
        return cur_P, cur_cond_P, new_distance_nn, new_neighbor_nn

    def _finetune_tsne(self, P, ex_ratio, degrees_of_freedom, n_samples, X_embedded, skip_num_points=0):
        params = X_embedded.ravel()
        ex_iter = self.n_iter_ + int(ex_ratio * self.finetune_iter) + 1
        cur_finetune_iter = self.n_iter_ + self.finetune_iter
        opt_args = {
            "it": self.n_iter_,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self._learning_rate,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points, angle=self.angle, verbose=self.verbose,
                           num_threads=_openmp_effective_n_threads()),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self.n_iter_without_progress,
            "n_iter": ex_iter,
            "momentum": 0.5,
        }

        P *= self.early_exaggeration
        params, kl_divergence, it = _gradient_descent(_kl_divergence_bh, params, **opt_args)
        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations with early exaggeration: %f" % (it + 1, kl_divergence))

        P /= self.early_exaggeration

        if cur_finetune_iter - ex_iter > 0:
            opt_args["n_iter"] = cur_finetune_iter
            opt_args["it"] = it + 1
            opt_args["momentum"] = 0.8
            opt_args["n_iter_without_progress"] = self.n_iter_without_progress
            params, kl_divergence, it = _gradient_descent(_kl_divergence_bh, params, **opt_args)

        self.n_iter_ = it

        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations: %f" % (it + 1, kl_divergence))

        new_x_embeddings = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return new_x_embeddings

    def ending(self):
        pass


if __name__ == "__main__":

    # Generate test data
    df = h5py.File("../../../Data/H5 Data/isolet_subset.h5", "r")
    x = np.array(df['x'], dtype=float)
    y = np.array(df['y'], dtype=int)
    initial_labels = [15, 24, 25]

    initial_indices = []
    for item in initial_labels:
        cur_indices = np.argwhere(y == item).squeeze()
        initial_indices.extend(cur_indices)

    test_data_high = x[initial_indices]
    test_class = y[initial_indices]
    after_indices = np.setdiff1d(np.arange(0, len(y), 1), initial_indices)
    np.random.shuffle(after_indices)

    streaming_data_high = x[after_indices]
    streaming_class = y[after_indices]

    perplexity = 30
    tsne = atSNEModel(finetune_iter=300, perplexity=perplexity, n_components=2, init='pca', n_iter=1000, verbose=2,
                      rho=0.99)

    low_dim_embeds = tsne.fit_transform(test_data_high)

    draw_projections(low_dim_embeds, test_class)

    num_per_time = 100
    timestep_counts = int(len(after_indices) / num_per_time)
    total_label = test_class
    for t in range(timestep_counts):
        print("处理第{}个时间步！".format(t + 1))
        cur_s_data = streaming_data_high[t * num_per_time:min(len(streaming_data_high) - 1, (t + 1) * num_per_time)]
        cur_s_label = streaming_class[t * num_per_time:min(len(streaming_class) - 1, (t + 1) * num_per_time)]
        embeddings = tsne.fit_new_data(cur_s_data)

        total_label = np.concatenate([total_label, cur_s_label])

        draw_projections(embeddings, total_label)
