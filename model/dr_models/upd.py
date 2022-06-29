import time
import h5py
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from experiments.experiment import position_vis
from sklearn.utils.extmath import randomized_svd


tol = 1.e-6  # zero tolerance
epsilon = 1e-7


def cal_dist(data):
    data_matrix = np.repeat(np.expand_dims(data, axis=0), axis=0, repeats=data.shape[0])
    dists = np.linalg.norm(data_matrix - np.transpose(data_matrix, axes=(1, 0, 2)), axis=-1)
    return dists


class ForceScheme:
    def __init__(self, n_iter=50, delta_frac=8, verbose=False):
        self.n_iter = n_iter
        self.delta_frac = delta_frac
        self.verbose = verbose

    def fit_transform(self, data, labels=None):
        start_time = time.time()
        if self.verbose:
            print("Start Project...")

        n_samples, dims = data.shape  # number of instances, dimension of the data

        z = np.random.random((n_samples, 2))  # random initialization

        dists = cal_dist(data)

        idx = np.random.permutation(n_samples)

        for k in range(self.n_iter):
            # for each x'
            if self.verbose:
                print("Iteration: ", k)
            for i in range(n_samples):
                inst1 = idx[i]
                # for each q' != x'
                for j in range(n_samples):
                    inst2 = idx[j]
                    if inst1 != inst2:
                        # computes direction v
                        v = z[inst2] - z[inst1]
                        dists_2 = np.hypot(v[0], v[1])
                        if dists_2 < tol:
                            dists_2 = tol
                        delta = (dists[inst1][inst2] - dists_2) / self.delta_frac
                        v /= dists_2
                        # move q' = Y[j] in the direction of v by a fraction of delta
                        z[inst2] += delta * v
        if self.verbose:
            print("Done. Elapsed time:", time.time() - start_time, "s.")
        if labels is not None:
            position_vis(labels, None, z)
        return z


def _cal_B(dists, n_samples):
    sum_ = np.sum(dists, axis=1) / n_samples
    Di = np.repeat(sum_[:, np.newaxis], n_samples, axis=1)
    Dj = np.repeat(sum_[np.newaxis, :], n_samples, axis=0)
    Dij = np.sum(dists) / (n_samples ** 2) * np.ones([n_samples, n_samples])
    B = (Di + Dj - dists - Dij) / 2
    return B


def _cal_X(dists, eigen_val, eigen_vector, n_samples):
    # 1*n
    dists_ga_col = np.mean(dists, axis=1)
    # n * n
    D_2 = dists.T - np.repeat(np.expand_dims(dists_ga_col, axis=0), axis=0, repeats=n_samples)
    # r * n
    normed_eigen_vector = eigen_vector.T / np.expand_dims(np.sqrt(eigen_val), axis=1)
    # r * n
    X = 0.5 * np.matmul(normed_eigen_vector, D_2).T
    return X


def _sampling(data):
    n_samples = data.shape[0]
    km = KMeans(n_clusters=int(np.sqrt(n_samples)))
    km.fit(data)
    centroids = km.cluster_centers_
    # len(centroids) * n
    dists = np.linalg.norm(np.repeat(np.expand_dims(data, axis=0), axis=0, repeats=centroids.shape[0]) - \
                           np.repeat(np.expand_dims(centroids, axis=1), axis=1, repeats=n_samples), axis=-1)

    sampled_indices = np.argsort(dists, axis=-1)[:, 0]
    sampled_x = data[sampled_indices]
    return sampled_x


def _do_svd(B, eta):
    eigen_val, eigen_vector = np.linalg.eigh(B)
    sort_indices = np.argsort(-eigen_val)
    ev_num = 0
    acc_val = 0
    eigen_val_sum = np.sum(eigen_val)
    for ev_num in range(len(eigen_val)):
        if acc_val >= eta:
            break
        acc_val += eigen_val[sort_indices[ev_num]] / eigen_val_sum

    ev_num = max(2, ev_num)
    eigen_val = eigen_val[sort_indices[:ev_num]]  # 降序排列的特征值
    eigen_vector = eigen_vector[:, sort_indices[:ev_num]]  # 降序排列的特征值对应的特征向量
    return eigen_val, eigen_vector


def _recover_origin_data(dists, eta):
    """
    根据距离矩阵恢复原始的高维数据
    :param dists: 数据的距离矩阵，欧氏距离平方
    :return: 高维数据
    """
    dists **= 2
    n_samples = dists.shape[0]
    B = _cal_B(dists, n_samples)

    eigen_val, eigen_vector = _do_svd(B, eta)
    X = _cal_X(dists, eigen_val, eigen_vector, n_samples)
    return X


class UPDis:
    def __init__(self, eta=0.99):
        self.eta = eta
        self.control_points = None
        self.control_embeddings = None

    def fit_transform_cntp_free(self, dists, labels=None):
        # 1. 使用C-MDS自动生成control points
        origin_data = _recover_origin_data(dists, self.eta)
        control_points = _sampling(origin_data)
        # 2. 投影
        embeddings = self.real_project(control_points, origin_data)
        if labels is not None:
            position_vis(labels, None, embeddings)
        return embeddings

    def real_project(self, control_points, project_points):
        reducer = ForceScheme()
        self.control_points = control_points
        self.control_embeddings = reducer.fit_transform(control_points)
        return self._project(project_points)

    def _project(self, project_x):
        weights = cdist(project_x, self.control_points)
        weights = 1.0 / (weights + epsilon)
        project_num = project_x.shape[0]
        embeddings = np.zeros((project_num, 2))

        for i in range(project_num):
            alpha = np.sum(weights[i])
            x_tilde = np.dot(self.control_points.T, weights[i].T) / alpha
            y_tilde = np.dot(self.control_embeddings.T, weights[i].T) / alpha
            x_hat = self.control_points - x_tilde
            y_hat = self.control_embeddings - y_tilde
            M = _cal_M(weights[i], x_hat, y_hat)
            embeddings[i] = np.dot((project_x[i] - x_tilde), M) + y_tilde

        return embeddings


def _cal_M(weight, x_hat, y_hat):
    D = np.diag(np.sqrt(weight))
    A = np.dot(D, x_hat)
    B = np.dot(D, y_hat)
    U, s, V = randomized_svd(np.dot(A.T, B), n_components=2, random_state=None)
    M = np.dot(U, V)
    return M


class UPDis4Streaming(UPDis):
    def __init__(self, eta=0.99):
        UPDis.__init__(self, eta)
        self.ctp2ctp_dists = None
        self.normed_eigen_vector = None

    def fit_transform(self, dists2ctp, ctp2ctp_dists):
        """
        :param dists2ctp: 所有点到control points的二范数距离，n*m，n表示所有点的个数，m表示control points的个数
        :param ctp2ctp_dists: control points之间的二范数距离，m*m
        :return:
        """
        # 只使用control points之间的距离来还原其他点的高维坐标
        square_ctp2ctp_dists = ctp2ctp_dists ** 2
        self.ctp2ctp_dists = square_ctp2ctp_dists

        n_samples = square_ctp2ctp_dists.shape[0]
        B = _cal_B(square_ctp2ctp_dists, n_samples)
        # 1*r, k*r
        eigen_val, eigen_vector = _do_svd(B, self.eta)
        control_points = _cal_X(square_ctp2ctp_dists, eigen_val, eigen_vector, n_samples)
        self.control_points = control_points

        # r * k
        self.normed_eigen_vector = eigen_vector.T / np.expand_dims(np.sqrt(eigen_val), axis=1)

        reducer = ForceScheme()
        control_embeddings = reducer.fit_transform(control_points)
        self.control_embeddings = control_embeddings

        return self.reuse_project(dists2ctp)

    def reuse_project(self, dists2ctp):
        # 1*k
        square_dists2ctp = dists2ctp ** 2
        dists_ga_col = np.mean(self.ctp2ctp_dists, axis=1)

        project_num = square_dists2ctp.shape[0]
        embeddings = np.zeros((project_num, 2))
        for i in range(project_num):
            cur_dists = np.expand_dims(square_dists2ctp[i].T - dists_ga_col, axis=0)
            project_x = 0.5 * np.matmul(self.normed_eigen_vector, cur_dists.T).T

            weight = cdist(project_x, self.control_points).squeeze()
            weight = 1.0 / (weight + epsilon)
            alpha = np.sum(weight)

            x_tilde = np.dot(self.control_points.T, weight.T) / alpha
            y_tilde = np.dot(self.control_embeddings.T, weight.T) / alpha
            x_hat = self.control_points - x_tilde
            y_hat = self.control_embeddings - y_tilde
            M = _cal_M(weight, x_hat, y_hat)
            embeddings[i] = np.dot((project_x - x_tilde), M) + y_tilde
        return embeddings


if __name__ == '__main__':
    hf = h5py.File("../../../../Data/H5 Data/wethers.h5", "r")
    x = np.array(hf['x'], dtype=float)
    y = np.array(hf['y'], dtype=int)

    # fs = ForceScheme()
    fs = UPDis()
    fs.fit_transform_cntp_free(cal_dist(x), y)
