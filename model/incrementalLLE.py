import h5py
import scipy.linalg
from scipy.linalg import eigh, schur
from scipy.sparse.linalg import eigsh
from sklearn.manifold import LocallyLinearEmbedding
import numpy as np
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph, barycenter_weights
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import eye
from sklearn.utils._arpack import _init_arpack_v0
from model.scdr.dependencies.experiment import position_vis
from utils.nn_utils import compute_knn_graph
from model.incrementalLE import kNNBasedIncrementalMethods


class IncrementalLLE(LocallyLinearEmbedding, kNNBasedIncrementalMethods):
    def __init__(self, train_num, n_components, n_neighbors, iter_num=50):
        LocallyLinearEmbedding.__init__(self, n_neighbors=n_neighbors, n_components=n_components)
        kNNBasedIncrementalMethods.__init__(self, train_num, n_components, n_neighbors)
        self.weight_matrix = None
        self.eigen_solver = "auto"
        self.iter_num = iter_num
        # 只在第一次训练时用到
        self.eigen_values = None
        self.eigen_vectors = None
        self.k_skip = 1

    def _first_train(self, train_data):
        self.pre_embeddings = self.fit_transform(train_data)
        knn_indices, knn_dists = compute_knn_graph(train_data, None, self.n_neighbors, None)
        self.knn_manager.add_new_kNN(knn_indices, knn_dists)
        return self.pre_embeddings

    def _incremental_embedding(self, new_data):
        new_knn_indices, new_knn_dists, neighbor_changed_indices = self._update_kNN(new_data)

        self._update_weight_matrix(new_data, neighbor_changed_indices)

        self.pre_embeddings = self._embedding_new_data_RayleighRitz(new_data.shape[0])
        return self.pre_embeddings

    def _update_weight_matrix(self, new_data, neighbor_changed_indices):
        new_data_num = new_data.shape[0]
        neighbor_changed_data = self.stream_dataset._total_data[neighbor_changed_indices]
        indices = self.knn_manager.knn_indices[neighbor_changed_indices]
        changed_weights = barycenter_weights(neighbor_changed_data, self.stream_dataset._total_data, indices)
        pre_data_num = self.weight_matrix.shape[0]
        self.weight_matrix = np.concatenate([self.weight_matrix, np.zeros((pre_data_num, new_data_num))], axis=1)
        self.weight_matrix = np.concatenate([self.weight_matrix, np.zeros((new_data_num, new_data_num + pre_data_num))], axis=0)
        self.weight_matrix[neighbor_changed_indices] *= 0

        for i, item in enumerate(neighbor_changed_indices):
            self.weight_matrix[item, self.knn_manager.knn_indices[item]] = changed_weights[i]

        new_data_weights = barycenter_weights(new_data, self.stream_dataset._total_data,
                                              self.knn_manager.knn_indices[pre_data_num:])
        for i in range(new_data_num):
            index = pre_data_num + i
            self.weight_matrix[index, self.knn_manager.knn_indices[index]] = new_data_weights[i]

    def _embedding_new_data_RayleighRitz(self, new_data_num):
        Q0 = np.concatenate([self.eigen_vectors, np.zeros((new_data_num, self.n_components + self.k_skip))])

        for i in range(self.iter_num):
            T = self.weight_matrix * Q0
            Q_bar, R = np.linalg.qr(T)
            T_star = Q_bar[:, :self.n_components+self.k_skip].T * self.weight_matrix * \
                     Q_bar[:, :self.n_components+self.k_skip]
            _, U = schur(T_star)
            Q0 = Q_bar[:, :self.k_skip+self.n_components] * U

        return np.array(Q0[:, self.k_skip:])

    def _fit_transform(self, X):
        neighbors = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        neighbors.fit(X)
        is_m_sparse = (self.eigen_solver != 'dense')

        W = barycenter_kneighbors_graph(neighbors, n_neighbors=self.n_neighbors, )
        self.weight_matrix = W.todense()

        # compute M = (I-W)'(I-W)
        if is_m_sparse:
            M = eye(*W.shape, format=W.format) - W
            M = (M.T * M).tocsr()
        else:
            M = (W.T * W - W.T - W).toarray()
            M.flat[::M.shape[0] + 1] += 1  # W = W - I = W - I

        self.eigen_vectors, self.eigen_values = self._my_null_space(M)
        self.embedding_ = self.eigen_vectors[:, self.k_skip:]

    def _my_null_space(self, M):
        k = self.n_components
        eigen_solver = self.eigen_solver
        if eigen_solver == "auto":
            if M.shape[0] > 200 and k + self.k_skip < 10:
                eigen_solver = "arpack"
            else:
                eigen_solver = "dense"

        if eigen_solver == "arpack":
            v0 = _init_arpack_v0(M.shape[0], self.random_state)
            try:
                eigen_values, eigen_vectors = eigsh(
                    M, k + self.k_skip, sigma=0.0, tol=self.tol, maxiter=self.max_iter, v0=v0
                )
            except RuntimeError as e:
                raise ValueError(
                    "Error in determining null-space with ARPACK. Error message: "
                    "'%s'. Note that eigen_solver='arpack' can fail when the "
                    "weight matrix is singular or otherwise ill-behaved. In that "
                    "case, eigen_solver='dense' is recommended. See online "
                    "documentation for more information." % e
                ) from e

            return eigen_vectors, eigen_values
        elif eigen_solver == "dense":
            if hasattr(M, "toarray"):
                M = M.toarray()
            eigen_values, eigen_vectors = eigh(
                M, eigvals=(self.k_skip, k + self.k_skip - 1), overwrite_a=True
            )
            index = np.argsort(np.abs(eigen_values))
            return eigen_vectors[:, index], eigen_values
        else:
            raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)


if __name__ == '__main__':
    with h5py.File("../../../Data/H5 Data/Anuran Calls_10c.h5", "r") as hf:
        X = np.array(hf['x'])
        Y = np.array(hf['y'])

    train_num = 2000
    train_data = X[:train_num]
    train_labels = Y[:train_num]

    ile = IncrementalLLE(train_num, 2, 10)

    first_embeddings = ile.fit_new_data(train_data)
    position_vis(train_labels, None, first_embeddings, "first")

    second_embeddings = ile.fit_new_data(X[train_num:train_num + 1000])
    position_vis(Y[train_num:train_num + 1000], None, second_embeddings[train_num:], "second new")
    position_vis(train_labels, None, second_embeddings[:train_num], "second pre")
    position_vis(Y[:train_num + 1000], None, second_embeddings, "second whole")
