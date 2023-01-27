# 需要更新 self.pre_cluster_num， self.isomap_list， self.transformation_info_list， self.global_embedding_mean，self.cluster_indices，self.pre_embeddings
import copy
from threading import Thread
import numpy as np

from model.parallel_ine import INEChangeDetector
from model.stream_isomap import SIsomapPlus, _extract_labels
from utils.nn_utils import compute_knn_graph


class ParallelSIsomapP(SIsomapPlus):
    def __init__(self, pattern_data_queue, model_update_queue, model_return_queue, replace_model_queue, train_num,
                 n_components, n_neighbors, epsilon=0.25):
        SIsomapPlus.__init__(self, train_num, n_components, n_neighbors, epsilon)
        self._pattern_data_queue = pattern_data_queue
        self._model_return_queue = model_return_queue
        self._replace_model_queue = replace_model_queue
        self._model_update_queue = model_update_queue
        self.pattern_detector = INEChangeDetector(pattern_data_queue, model_update_queue, replace_model_queue)
        self.model_updater = None

        self._get_new_model = False
        self._new_embeddings = None
        self._new_isomap_list = None
        self._new_transformation_info_list = None
        self._new_global_embedding_mean = None
        self._new_cluster_indices = None
        self._new_cluster_num = None

    def _first_train(self, train_data):
        self.pre_embeddings = super()._first_train(train_data)
        self.pattern_detector.start()
        self._pattern_data_queue.put([train_data, False, train_data, ])
        self.model_updater = SIsomapUpdater(self._model_update_queue, self._model_return_queue, self.initial_train_num,
                                            self.n_components, self.n_neighbors)
        self.model_updater.start()

    def _incremental_embedding(self, new_data):

        if not self._replace_model_queue.empty():
            replace_model = self._replace_model_queue.get()
            if replace_model:
                print("replace model!")

                if not self._get_new_model:
                    self._get_new_model_info()

                replace_data_num = self._new_embeddings.shape[0]
                unreplace_data_num = self.pre_embeddings.shape[0] - replace_data_num
                self.pre_embeddings[:replace_data_num] = self._new_embeddings
                self.isomap_list = self._new_isomap_list
                self.transformation_info_list = self._new_transformation_info_list
                self.pre_cluster_num = self._new_cluster_num
                self.global_embedding_mean = self._new_global_embedding_mean
                self.cluster_indices = self._new_cluster_indices

                cur_cluster_idx = np.zeros(self.pre_cluster_num)
                arr_data_cluster = np.array(self._data_cluster)
                for i in range(self.pre_cluster_num):
                    cur_cluster_idx[i] = np.mean(arr_data_cluster[self._new_cluster_indices[i]])

                dists = np.abs(np.repeat(arr_data_cluster[replace_data_num:][:, np.newaxis], axis=1,
                                         repeats=self.pre_cluster_num) -
                               np.repeat(cur_cluster_idx[np.newaxis, :], axis=0, repeats=unreplace_data_num))
                corr_cluster_idx = np.argmin(dists, axis=1)
                for i in range(unreplace_data_num):
                    c_id = corr_cluster_idx[i]
                    self.cluster_indices[c_id].append(replace_data_num + i)
                    # self.isomap_list[c_id].merge_new_data_info(new_data,
                    #                                            self.pre_embeddings[replace_data_num + i][np.newaxis, :],
                    #                                            self._geodesic_dists[replace_data_num +
                    #                                                                 i - self.initial_train_num])

                self._get_new_model = False

        if not self._model_return_queue.empty():
            self._get_new_model_info()

        self._pattern_data_queue.put([new_data, False, self.stream_dataset.get_total_data(), ])

        return super()._incremental_embedding(new_data)

    def _get_new_model_info(self):
        self._new_embeddings, self._new_isomap_list, self._new_transformation_info_list, \
            self._new_global_embedding_mean, self._new_cluster_indices = self._model_return_queue.get()
        self._new_cluster_num = len(self._new_cluster_indices)
        self._get_new_model = True


class SIsomapUpdater(Thread, SIsomapPlus):
    def __init__(self, model_update_queue, model_return_queue, train_num, n_components, n_neighbors, epsilon=0.25):
        Thread.__init__(self, name="SIsomapUpdater")
        SIsomapPlus.__init__(self, train_num, n_components, n_neighbors, epsilon)
        self._model_update_queue = model_update_queue
        self._model_return_queue = model_return_queue

    def run(self) -> None:
        while True:
            stop_flag, data = self._model_update_queue.get()

            if stop_flag:
                break

            knn_indices, knn_dists = compute_knn_graph(data, None, self.n_neighbors, None)
            # self.knn_manager.add_new_kNN(knn_indices, knn_dists)
            self.cluster_indices = self._find_clusters(data, knn_indices)
            local_embeddings_list = self._local_embedding(data)
            seq_cluster_indices, seq_labels, seq_local_embeddings = \
                _extract_labels(self.cluster_indices, local_embeddings_list)

            self.pre_cluster_num = len(self.cluster_indices)
            if self.pre_cluster_num > 1:
                global_embeddings, support_set_indices = self._global_embedding(data)
                transformed_embeddings = self._euclidean_transformation(support_set_indices, global_embeddings,
                                                                        seq_local_embeddings)
                embeddings = transformed_embeddings
            else:
                embeddings = seq_local_embeddings

            self._model_return_queue.put([embeddings, copy.copy(self.isomap_list),
                                          copy.copy(self.transformation_info_list), copy.copy(self.global_embedding_mean),
                                          copy.copy(self.cluster_indices)])
            self.isomap_list = []
            self.transformation_info_list = []
            self.global_embedding_mean = []
            self.cluster_indices = None
