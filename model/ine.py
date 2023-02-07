import time

import h5py
import numpy as np
import scipy.optimize
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from model.incrementalLE import kNNBasedIncrementalMethods
from sklearn.manifold import TSNE
from model.scdr.dependencies.experiment import position_vis
from utils.logger import InfoLogger
from utils.loss_grads import tsne_grad
from utils.nn_utils import compute_knn_graph


def _select_min_loss_one(candidate_embeddings, neighbors_embeddings, high_probabilities, k):
    # [G*G, n]
    dists = cdist(candidate_embeddings, neighbors_embeddings)
    tmp_prob = 1 / (1 + dists ** 2) ** (0.5 + k/10)
    q = tmp_prob / np.expand_dims(np.sum(tmp_prob, axis=1), axis=1)
    high_prob_matrix = np.repeat(high_probabilities, candidate_embeddings.shape[0], axis=0)
    # [G*G]
    loss_list = -np.sum(high_prob_matrix * np.log(q), axis=1)
    # print("prev min loss:", np.min(loss_list))
    return candidate_embeddings[np.argmin(loss_list)]


class INEModel(kNNBasedIncrementalMethods, TSNE):
    def __init__(self, train_num, n_components, n_neighbors, iter_num=100, grid_num=27, desired_perplexity=3,
                 init="random", window_size=2000):
        kNNBasedIncrementalMethods.__init__(self, train_num, n_components, n_neighbors, True, window_size)
        TSNE.__init__(self, n_components, perplexity=n_neighbors)
        self.init = init
        self.desired_perplexity = desired_perplexity
        self.iter_num = iter_num
        self.grid_num = grid_num
        self.condition_P = None
        self._learning_rate = 200.0
        self._update_thresh = 5
        self._k = 5
        self._window_size = window_size

    def _slide_window(self):
        out_num = super()._slide_window()
        if out_num <= 0:
            return
        self.condition_P = self.condition_P[out_num:, :]

    def _first_train(self, train_data):
        self.pre_embeddings = self.fit_transform(train_data)
        knn_indices, knn_dists = compute_knn_graph(train_data, None, self.n_neighbors, None)
        self.knn_manager.add_new_kNN(knn_indices, knn_dists)
        return self.pre_embeddings

    def _incremental_embedding(self, new_data):
        # 一次只处理一个数据

        new_data = np.reshape(new_data, (1, -1))

        knn_indices, knn_dists, dists = self._cal_new_data_kNN(new_data, include_self=False)
        knn_indices = knn_indices.squeeze()

        new_data_prob = self._cal_new_data_probability(knn_dists.astype(np.float32, copy=False))

        initial_embedding = self._initialize_new_data_embedding(new_data_prob, knn_indices)
        # print("initial", initial_embedding)
        self.pre_embeddings = self._optimize_new_data_embedding(knn_indices, initial_embedding, new_data_prob)
        # print("after", self.pre_embeddings[-1])
        return self.pre_embeddings

    def _cal_new_data_probability(self, dists):
        # conditional_P = _binary_search_perplexity(dists**2, self.desired_perplexity, False)
        conditional_P = search_prob(dists**2, perplexity=self.desired_perplexity)
        self.condition_P = np.concatenate([self.condition_P, conditional_P], axis=0)
        return conditional_P

    def _initialize_new_data_embedding(self, high_prob, new_knn_indices):
        candidate_embeddings = self._generate_candidate_embeddings()
        initial_embeddings = _select_min_loss_one(candidate_embeddings, self.pre_embeddings[new_knn_indices],
                                                  high_prob, self._k)

        return initial_embeddings

    def _optimize_new_data_embedding(self, new_data_knn_indices, initial_embedding, new_data_prob):
        def loss_func(embeddings, high_prob, neighbor_embeddings, k):
            similarities = 1 / (1 + cdist(embeddings[np.newaxis, :], neighbor_embeddings) ** 2) ** (0.5 + k/10)
            normed_similarities = similarities / np.expand_dims(np.sum(similarities, axis=1), axis=1)
            return -np.sum(high_prob * np.log(normed_similarities))

        res = scipy.optimize.minimize(loss_func, initial_embedding, method="BFGS", jac=tsne_grad,
                                      args=(new_data_prob, self.pre_embeddings[new_data_knn_indices], self._k),
                                      options={'gtol': 1e-6, 'disp': False})
        # print("opt x:", res.x)
        if np.abs(res.x[0] - initial_embedding[0]) > self._update_thresh \
                or np.abs(res.x[1] - initial_embedding[1]) > self._update_thresh:
            # print("initial:", initial_embedding)
            # print(new_data_prob)
            new_embeddings = initial_embedding[np.newaxis, :]
        else:
            new_embeddings = res.x[np.newaxis, :]
        total_embeddings = np.concatenate([self.pre_embeddings, new_embeddings], axis=0)
        return total_embeddings

    def _generate_candidate_embeddings(self):
        x_min, y_min = np.min(self.pre_embeddings, axis=0)
        x_max, y_max = np.max(self.pre_embeddings, axis=0)
        # print("x scale: {} ~ {}".format(x_min, x_max))
        # print("y scale: {} ~ {}".format(y_min, y_max))
        x_grid_list = np.linspace(x_min, x_max, self.grid_num)
        y_grid_list = np.linspace(y_min, y_max, self.grid_num)

        x_grid_list_ravel = np.reshape(np.repeat(np.expand_dims(x_grid_list, axis=1), self.grid_num, 1), (-1, 1))
        y_grid_list_ravel = np.reshape(np.repeat(np.expand_dims(y_grid_list, axis=0), self.grid_num, 0), (-1, 1))
        candidate_embeddings = np.concatenate([x_grid_list_ravel, y_grid_list_ravel], axis=1)
        return candidate_embeddings

    def _fit(self, X, skip_num_points=0):
        n_samples = X.shape[0]

        neighbors_nn = None
        if self.method == "exact":
            raise RuntimeError("Only support Barnes Hut t-SNE")
        else:
            n_neighbors = min(n_samples - 1, int(self.perplexity))
            # Find the nearest neighbors for every point
            knn = NearestNeighbors(
                algorithm="auto",
                n_jobs=self.n_jobs,
                n_neighbors=n_neighbors,
                metric=self.metric,
            )
            knn.fit(X)
            distances_nn = knn.kneighbors_graph(mode="distance")
            del knn

            if self.square_distances is True or self.metric == "euclidean":
                distances_nn.data **= 2

            P, conditional_P = my_joint_probabilities_nn(distances_nn, self.perplexity)
            self.condition_P = conditional_P

        if self.init == "pca":
            pca = PCA(
                n_components=self.n_components,
                svd_solver="randomized",
                random_state=self.random_state,
            )
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
        elif self.init == "random":
            random_state = check_random_state(self.random_state)
            X_embedded = 1e-4 * random_state.randn(n_samples, self.n_components).astype(
                np.float32
            )
        else:
            raise ValueError("'init' must be 'pca', 'random', or a numpy array")

        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._tsne(
            P,
            degrees_of_freedom,
            n_samples,
            X_embedded=X_embedded,
            neighbors=neighbors_nn,
            skip_num_points=skip_num_points,
        )


def my_joint_probabilities_nn(distances, desired_perplexity):
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    conditional_P = _binary_search_perplexity(
        distances_data, desired_perplexity, False
    )
    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"

    P = csr_matrix(
        (conditional_P.ravel(), distances.indices, distances.indptr),
        shape=(n_samples, n_samples),
    )
    P = P + P.T

    sum_P = np.maximum(P.sum(), np.finfo(np.double).eps)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    return P, conditional_P


def cal_perplexity(dist, idx=0, beta=1.0):
    # '''计算perplexity, D是距离向量，
    # idx指dist中自己与自己距离的位置，beta是高斯分布参数
    # 这里的perp仅计算了熵，方便计算
    # '''
    prob = np.exp(-dist * beta)
    # 设置自身prob为0
    # prob[idx] = 0
    sum_prob = np.sum(prob)
    if sum_prob == 0:
        prob = np.maximum(prob, 1e-12)
        perplexity = -12
    else:
        prob /= sum_prob
        perplexity = 0
        for pj in prob:
            if pj != 0:
                perplexity += -pj * np.log(pj)
    # 困惑度和pi\j的概率分布
    return perplexity, prob


def search_prob(distances, tol=1e-5, perplexity=30.0, debug=False):
    # '''二分搜索寻找beta,并计算pairwise的probability
    # '''
    # 初始化参数

    if debug:
        InfoLogger.info("Computing pairwise distances...")
    (n, d) = distances.shape

    pair_prob = np.zeros_like(distances)
    beta = np.ones((n, 1))
    # 取log，方便后续计算
    base_perplexity = np.log(perplexity)

    for i in range(n):
        if debug and i % 10 == 0:
            InfoLogger.info("Computing pair_prob for point %s of %s ..." % (i, n))

        beta_min = -np.inf
        beta_max = np.inf
        # dist[i]需要换不能是所有点
        perplexity, cur_prob = cal_perplexity(distances[i], i, beta[i])

        # 二分搜索,寻找最佳sigma下的prob
        perplexity_diff = perplexity - base_perplexity
        tries = 0
        while np.abs(perplexity_diff) > tol and tries < 50:
            if perplexity_diff > 0:
                beta_min = beta[i].copy()
                if beta_max == np.inf or beta_max == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + beta_max) / 2
            else:
                beta_max = beta[i].copy()
                if beta_min == np.inf or beta_min == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + beta_min) / 2

            # 更新perb,prob值
            perplexity, cur_prob = cal_perplexity(distances[i], i, beta[i])
            perplexity_diff = perplexity - base_perplexity
            tries = tries + 1
        # 记录prob值
        pair_prob[i, ] = cur_prob
    if debug:
        InfoLogger.info("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    # 每个点对其他点的条件概率分布pi\j

    return pair_prob


if __name__ == '__main__':
    with h5py.File("../../../Data/H5 Data/food.h5", "r") as hf:
        X = np.array(hf['x'])
        Y = np.array(hf['y'])

    train_num = 1000
    train_data = X[:train_num]
    train_labels = Y[:train_num]

    ile = INEModel(train_num, 2, 10)

    first_embeddings = ile.fit_new_data(train_data)
    position_vis(train_labels, None, first_embeddings, "first")

    second_embeddings = ile.fit_new_data(X[train_num:train_num + 1000])
    position_vis(Y[train_num:train_num + 1000], None, second_embeddings[train_num:], "second new")
    position_vis(train_labels, None, second_embeddings[:train_num], "second pre")
    position_vis(Y[:train_num + 1000], None, second_embeddings, "second whole")
