import warnings

import h5py
import numpy as np
import skdim
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from skdim import get_nn
from skdim.id import lPCA
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import Isomap, MDS
from sklearn.utils.validation import check_array, check_is_fitted
from model.incrementalLE import kNNBasedIncrementalMethods
from model.scdr.dependencies.experiment import position_vis
from utils.nn_utils import compute_knn_graph


class SIsomap(kNNBasedIncrementalMethods, Isomap):
    def __init__(self, train_num, n_components, n_neighbors):
        Isomap.__init__(self, n_neighbors=n_neighbors, n_components=n_components)
        kNNBasedIncrementalMethods.__init__(self, train_num, n_components, n_neighbors, single=True)
        self.G = None
        self.newest_embeddings = None

    def _first_train(self, train_data):
        self.pre_embeddings = self.fit_transform(train_data)
        self.G = self.dist_matrix_ ** 2
        self.G *= -0.5
        self.newest_embeddings = self.pre_embeddings
        return self.pre_embeddings

    def _incremental_embedding(self, new_data):
        pre_data_num = self.G.shape[0]
        new_data = new_data[np.newaxis, :]
        knn_indices, knn_dists, dists = self._cal_new_data_kNN(new_data, include_self=False)
        geodesic_dists = self._cal_new_data_geodesic_distance(pre_data_num, knn_indices.squeeze(), knn_dists.squeeze())
        embedding = self._embedding_new_data(pre_data_num, geodesic_dists)
        self.newest_embeddings = embedding
        self._merge_meta_info(embedding, geodesic_dists)

        return self.pre_embeddings

    def _merge_meta_info(self, embedding, geodesic_dists):
        self.pre_embeddings = np.concatenate([self.pre_embeddings, embedding], axis=0)
        geodesic_dists_self = np.append(geodesic_dists, 0.0)
        self.G = np.concatenate([self.G, geodesic_dists[:, np.newaxis]], axis=1)
        self.G = np.concatenate([self.G, geodesic_dists_self[np.newaxis, :]], axis=0)

    def transform_new_data_s(self, new_data):
        pre_data_num = self.G.shape[0]
        knn_indices, knn_dists, dists = self._cal_new_data_kNN(new_data, include_self=True)
        geodesic_dists = self._cal_new_data_geodesic_distance(pre_data_num, knn_indices.squeeze(), knn_dists.squeeze())
        embedding = self._embedding_new_data(pre_data_num, geodesic_dists)
        return embedding, geodesic_dists

    def merge_new_data_info(self, data, embedding, geodesic_dists):
        self.stream_dataset.add_new_data(np.reshape(data, (1, -1)), None)
        self._merge_meta_info(embedding, geodesic_dists)

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
        c = 0.5 * (np.mean(new_data_g) * one_vector - new_data_g - np.mean(self.G) * one_vector + np.mean(self.G,
                                                                                                          axis=1))
        # 2 * 1
        p = np.dot(np.dot(np.linalg.inv(np.dot(self.pre_embeddings.T, self.pre_embeddings)),
                          self.pre_embeddings.T), c[:, np.newaxis])
        y_star = np.concatenate([self.pre_embeddings, p.T], axis=0)
        embedding = p - np.mean(y_star, axis=0)[:, np.newaxis]
        return embedding.T


def _sim(components_1, components_2):
    n_1 = components_1.shape[0]
    n_2 = components_2.shape[0]
    N = np.zeros((n_1, n_2))

    # for i in range(n_1):
    #     for j in range(n_2):
    #         N[i][j] = np.sum(np.dot(components_1[i], components_2[j]))
    # return np.sqrt(np.linalg.det(np.dot(N, N.T)))

    n_3 = min(n_1, n_2)
    score = 0
    for i in range(n_3):
        score += np.abs(np.dot(components_1[i].T, components_2[i]))
    score /= n_3
    return score


def _extract_labels(cluster_indices, local_embeddings_list):
    seq_cluster_indices = []
    seq_labels = []
    seq_embeddings = []
    for i, item in enumerate(cluster_indices):
        seq_cluster_indices.extend(item)
        seq_labels.extend([i] * len(item))
        seq_embeddings.extend(local_embeddings_list[i])

    seq_cluster_indices = np.array(seq_cluster_indices, dtype=int)
    seq_labels = np.array(seq_labels, dtype=int)
    seq_embeddings = np.array(seq_embeddings, dtype=float)

    re_indices = np.argsort(seq_cluster_indices)
    seq_cluster_indices = seq_cluster_indices[re_indices]
    seq_labels = seq_labels[re_indices]
    seq_embeddings = seq_embeddings[re_indices]
    return seq_cluster_indices, seq_labels, seq_embeddings


class SIsomapPlus(kNNBasedIncrementalMethods):
    def __init__(self, train_num, n_components, n_neighbors, epsilon=0.25):
        kNNBasedIncrementalMethods.__init__(self, train_num, n_components, n_neighbors, single=True)
        self.epsilon = epsilon
        self.pre_cluster_num = 0
        self.isomap_list = []
        self.transformation_info_list = []
        self.cluster_indices = None
        self.global_embedding_mean = []
        self._data_cluster = []
        self._geodesic_dists = []

        self.cluster_real_cls = []
        self.predict_cls = []

    def _first_train(self, train_data):
        self.initial_train_num = train_data.shape[0]
        knn_indices, knn_dists = compute_knn_graph(train_data, None, self.n_neighbors, None)
        self.knn_manager.add_new_kNN(knn_indices, knn_dists)
        self.cluster_indices = self._find_clusters(train_data, knn_indices)
        local_embeddings_list = self._local_embedding(train_data)
        seq_cluster_indices, seq_labels, seq_local_embeddings = \
            _extract_labels(self.cluster_indices, local_embeddings_list)
        self._data_cluster = seq_labels.tolist()

        # position_vis(self.stream_dataset.total_label, None, seq_local_embeddings, "local embeddings")

        self.pre_cluster_num = len(self.cluster_indices)
        if self.pre_cluster_num > 1:
            global_embeddings, support_set_indices = self._global_embedding(train_data)
            transformed_embeddings = self._euclidean_transformation(support_set_indices, global_embeddings,
                                                                    seq_local_embeddings)
            self.pre_embeddings = transformed_embeddings
        else:
            self.pre_embeddings = seq_local_embeddings

        return self.pre_embeddings

    def _incremental_embedding(self, new_data):
        local_embedding_list, geodesic_dists_list = self._embedding2each_manifold(new_data)
        if self.pre_cluster_num > 1:
            global_embedding_list = self._transform2global_space(local_embedding_list)
            # TODO: 分类的精度非常低
            idx = self._select_best_manifold(global_embedding_list)
            embeddings = global_embedding_list[idx]
            # self.predict_cls.append(self.cluster_real_cls[idx])
        else:
            idx = 0
            embeddings = local_embedding_list[0]

        self._data_cluster.append(idx)
        self._geodesic_dists.append(geodesic_dists_list[idx])
        embeddings = embeddings[np.newaxis, :]
        self.isomap_list[idx].merge_new_data_info(new_data, embeddings, geodesic_dists_list[idx])
        self.pre_embeddings = np.concatenate([self.pre_embeddings, embeddings], axis=0)
        return self.pre_embeddings

    def _clustering(self, labels, main_components, cls_idx, knn_indices):
        unlabeled_list = np.where(labels == -1)[0]
        unlabeled_num = len(unlabeled_list)
        start_idx = unlabeled_list[np.random.randint(unlabeled_num)]
        unlabeled_num -= 1
        data_indices = [start_idx]
        old_indices = [start_idx]
        labels[start_idx] = cls_idx
        count = 1

        while count > 0:
            count = 0
            new_indices = []

            for item in old_indices:
                for jtem in knn_indices[item]:
                    if labels[jtem] == -1:
                        sim = _sim(main_components[item], main_components[jtem])
                        # print(sim)
                        if sim >= self.epsilon:
                            new_indices.append(jtem)
                            labels[jtem] = cls_idx
                            count += 1
                            unlabeled_num -= 1

            data_indices.extend(new_indices)
            old_indices = new_indices

        return data_indices, labels, unlabeled_num

    def _find_clusters(self, data, knn_indices):
        dimension_pw_, main_components = CustomizedLPCA().fit_transform_pw(data, knn_indices)
        n_samples = data.shape[0]
        idx = 1
        labels = np.ones(n_samples) * -1
        unlabeled_num = n_samples
        cluster_indices = []
        unsigned_indices = []

        while unlabeled_num > self.n_neighbors:
            pre_unlabeled_num = unlabeled_num
            cur_cls_data_indices, labels, unlabeled_num = self._clustering(labels, main_components, idx, knn_indices)
            # 该聚类数据规模太少，不将其视为一个聚类
            if pre_unlabeled_num - unlabeled_num < self.n_neighbors:
                labels[cur_cls_data_indices] = -2
                unsigned_indices.extend(cur_cls_data_indices)
            else:
                cluster_indices.append(cur_cls_data_indices)
                idx += 1

        unsigned_indices.extend(np.where(labels == -1)[0])
        cur_mean_dist = [np.mean(data[item], axis=0) for item in cluster_indices]
        unsigned_data = data[unsigned_indices]
        dists = cdist(unsigned_data, cur_mean_dist)
        closest_indices = np.argmin(dists, axis=1)
        for i, item in enumerate(unsigned_indices):
            labels[item] = closest_indices[i] + 1
            cluster_indices[int(closest_indices[i])].append(item)

        # print("cluster number:", idx)
        # for item in cluster_indices:
        #     labels = self.stream_dataset.get_total_label()[item]
        #     cls, counts = np.unique(labels, return_counts=True)
        #     idx = np.argmax(counts)
        #     self.cluster_real_cls.append(cls[idx])
        return cluster_indices

    # def _find_clusters_acc(self, data):
    #     cluster_indices = []
    #     unique_cls = np.unique(self.stream_dataset.total_label)
    #     for item in unique_cls:
    #         indices = np.argwhere(self.stream_dataset.total_label == item).squeeze()
    #         cluster_indices.append(indices.tolist())
    #     print("cluster num:", len(cluster_indices))
    #     return cluster_indices

    def _local_embedding(self, data):
        cluster_embedding_list = []
        for item in self.cluster_indices:
            isomap_embedder = SIsomap(len(item), self.n_components, self.n_neighbors)
            embeddings = isomap_embedder.fit_new_data(data[item])
            cluster_embedding_list.append(embeddings)
            self.isomap_list.append(isomap_embedder)

        return cluster_embedding_list

    def _global_embedding(self, data):
        cluster_num = len(self.cluster_indices)
        support_set = set()

        for i in range(cluster_num):
            for j in range(i + 1, cluster_num):
                c_dist = cdist(data[self.cluster_indices[i]], data[self.cluster_indices[j]])
                rows, cols = c_dist.shape
                sort_indices = np.argsort(c_dist, axis=None)

                farest_indices = sort_indices[-self.n_neighbors:]
                farest = [farest_indices // cols, farest_indices % cols]
                real_farest_indices = [np.array(self.cluster_indices[i])[farest[0]],
                                       np.array(self.cluster_indices[j])[farest[1]]]

                nearest_indices = sort_indices[:self.n_neighbors]
                nearest = [nearest_indices // cols, nearest_indices % cols]
                real_nearest_indices = [np.array(self.cluster_indices[i])[nearest[0]],
                                        np.array(self.cluster_indices[j])[nearest[1]]]

                cur_support_set = np.concatenate([real_nearest_indices, real_farest_indices], axis=1)

                support_set = support_set.union(np.ravel(cur_support_set))

        mds_embedder = MDS(self.n_components)
        global_embedding = mds_embedder.fit_transform(data[list(support_set)])

        return global_embedding, list(support_set)

    def _euclidean_transformation(self, support_indices, global_embeddings, local_embeddings, lam=0.05):
        cluster_num = len(self.cluster_indices)
        n_samples = local_embeddings.shape[0]
        transformed_embeddings = np.zeros((n_samples, self.n_components))
        for i in range(cluster_num):
            current_support_indices, indices_1, _ = np.intersect1d(support_indices, self.cluster_indices[i],
                                                                   return_indices=True)
            e = np.ones((1, len(indices_1)))
            A = np.concatenate([local_embeddings[current_support_indices].T, e], axis=0)
            RT = np.dot(global_embeddings[indices_1].T,
                        np.dot(A.T, np.linalg.inv((np.dot(A, A.T) + lam * np.identity(A.shape[0])))))
            R_i = np.linalg.qr(RT[:, :-1])[0]
            t_i = np.sum(global_embeddings[indices_1].T - np.dot(R_i, local_embeddings[current_support_indices].T),
                         axis=1) / len(indices_1)
            t_i = t_i[:, np.newaxis]

            self.transformation_info_list.append([R_i, t_i])
            cur_embeddings = np.dot(R_i, local_embeddings[self.cluster_indices[i]].T) + t_i
            transformed_embeddings[self.cluster_indices[i]] = cur_embeddings.T
            self.global_embedding_mean.append(np.mean(cur_embeddings.T, axis=0))

        return transformed_embeddings

    def _embedding2each_manifold(self, data):
        # print("pre_cluster_num", self.pre_cluster_num)
        embedding_list = np.zeros((self.pre_cluster_num, self.n_components))
        geodesic_dists_list = []
        for i, m_embedder in enumerate(self.isomap_list):
            embedding, geodesic_dists = m_embedder.transform_new_data_s(np.reshape(data, (1, -1)))
            embedding_list[i] = embedding
            geodesic_dists_list.append(geodesic_dists)
        return embedding_list, geodesic_dists_list

    def _transform2global_space(self, local_embedding_list):
        global_embedding_list = np.zeros((self.pre_cluster_num, self.n_components))
        # print(len(local_embedding_list), len(self.transformation_info_list))
        for i, item in enumerate(self.transformation_info_list):
            R_i, t_i = item
            embedding = np.dot(R_i, local_embedding_list[i].T)[:, np.newaxis] + t_i
            global_embedding_list[i] = embedding.T

        return global_embedding_list

    def _select_best_manifold(self, global_embedding_list):
        dists = np.linalg.norm(global_embedding_list - np.array(self.global_embedding_mean), axis=1)
        min_idx = np.argmin(dists, axis=None)

        manifold_data_num = len(self.cluster_indices[min_idx])
        self.global_embedding_mean[min_idx] = (manifold_data_num * self.global_embedding_mean[min_idx] +
                                               global_embedding_list[min_idx]) / (manifold_data_num + 1)
        self.cluster_indices[min_idx].append(self.stream_dataset.get_n_samples())
        return min_idx


class CustomizedLPCA(lPCA):
    def __init__(self, ver="FO", alphaRatio=0.05, alphaFO=0.05, alphaFan=10, betaFan=0.8, PFan=0.95, verbose=True):
        lPCA.__init__(self, ver, alphaRatio, alphaFO, alphaFan, betaFan, PFan, verbose, False)
        self.intrinsic_components = None

    def _pcaLocalDimEst(self, X):

        pca = PCA()
        pca.fit(X)
        self.explained_var_ = explained_var = pca.explained_variance_

        res = 0
        if self.ver == "FO":
            res = self._FO(explained_var)
        elif self.ver == "Fan":
            res = self._fan(explained_var)
        elif self.ver == "maxgap":
            res = self._maxgap(explained_var)
        elif self.ver == "ratio":
            res = self._ratio(explained_var)
        elif self.ver == "participation_ratio":
            res = self._participation_ratio(explained_var)
        elif self.ver == "Kaiser":
            res = self._Kaiser(explained_var)
        elif self.ver == "broken_stick":
            res = self._broken_stick(explained_var)

        self.intrinsic_components = pca.components_[:res[0]]
        return res

    def fit_transform_pw(self, X, precomputed_knn=None, smooth=False, n_neighbors=100, n_jobs=1):
        X = check_array(X, ensure_min_samples=n_neighbors + 1, ensure_min_features=2)

        if precomputed_knn is not None:
            knnidx = precomputed_knn
        else:
            _, knnidx = get_nn(X, k=n_neighbors, n_jobs=n_jobs)

        dimension_pw_ = []
        main_components_pw_ = []

        for item in knnidx:
            self.fit(X[item, :])
            dimension_pw_.append(self.dimension_)
            main_components_pw_.append(self.intrinsic_components)

        dimension_pw_ = np.array(dimension_pw_)
        main_components_pw_ = np.array(main_components_pw_)

        if smooth:
            dimension_pw_smooth_ = np.zeros(len(knnidx))
            for i, point_nn in enumerate(knnidx):
                dimension_pw_smooth_[i] = np.mean(
                    np.append(dimension_pw_[i], dimension_pw_[point_nn])
                )
            return dimension_pw_, dimension_pw_smooth_, main_components_pw_
        else:
            return dimension_pw_, main_components_pw_


if __name__ == '__main__':
    with h5py.File("../../../Data/H5 Data/food.h5", "r") as hf:
        X = np.array(hf['x'])
        Y = np.array(hf['y'])

    train_num = 2000
    oos_num = 1000
    train_data = X[:train_num]
    train_labels = Y[:train_num]

    ile = SIsomap(train_num, 2, 10)
    # ile = SIsomapPlus(train_num, 2, 10)

    first_embeddings = ile.fit_new_data(train_data, train_labels)
    position_vis(train_labels, None, first_embeddings, "first")

    second_embeddings = ile.fit_new_data(X[train_num:train_num + oos_num])
    position_vis(Y[train_num:train_num + oos_num], None, second_embeddings[train_num:], "second new")
    position_vis(train_labels, None, second_embeddings[:train_num], "second pre")
    position_vis(Y[:train_num + oos_num], None, second_embeddings, "second whole")

    # cls_acc = np.sum(np.array(ile.predict_cls, dtype=int) == Y[train_num:train_num + oos_num]) / oos_num
    # print("cls acc", cls_acc)
