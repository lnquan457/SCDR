# 需要更新 self.pre_cluster_num， self.isomap_list， self.transformation_info_list， self.global_embedding_mean，self.cluster_indices，self.pre_embeddings
import copy
from multiprocessing import Process
from threading import Thread
import numpy as np

from model.parallel_ine import INEChangeDetector
from model.stream_isomap import SIsomapPlus, _extract_labels
from utils.nn_utils import compute_knn_graph


class ParallelSIsomapP(SIsomapPlus):
    def __init__(self, pattern_data_queue, model_update_queue, model_return_queue, replace_model_queue,
                 update_finish_queue, train_num, n_components, n_neighbors, epsilon=0.25, window_size=2000):
        SIsomapPlus.__init__(self, train_num, n_components, n_neighbors, epsilon, window_size)
        self._pattern_data_queue = pattern_data_queue
        self._model_return_queue = model_return_queue
        self._replace_model_queue = replace_model_queue
        self._model_update_queue = model_update_queue
        self._update_finish_queue = update_finish_queue
        self.pattern_detector = INEChangeDetector(pattern_data_queue, model_update_queue, replace_model_queue, update_finish_queue)
        self.model_updater = None
        self._total_data_idx = 0
        self._newest_model_data_idx = 0

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
        self._pattern_data_queue.put([train_data, False, train_data, 0])
        self.model_updater = SIsomapUpdater(self._model_update_queue, self._model_return_queue,
                                            self._update_finish_queue, self.initial_train_num,
                                            self.n_components, self.n_neighbors)
        self.model_updater.start()
        self._total_data_idx = self.pre_embeddings.shape[0]

    def _incremental_embedding(self, new_data):
        self._total_data_idx += new_data.shape[0]
        if not self._replace_model_queue.empty():
            replace_model = self._replace_model_queue.get()
            if replace_model:
                print("replace model!")

                if not self._get_new_model:
                    self._get_new_model_info()

                out_num = self._replace_model(new_data.shape[0])

                self._get_new_model = False
                if out_num > 0:
                    self.check_embedding_validity(out_num)

        if not self._model_return_queue.empty():
            self._get_new_model_info()

        self._pattern_data_queue.put([new_data, False, self.stream_dataset.get_total_data(), self._total_data_idx])

        return super()._incremental_embedding(new_data)

    def _replace_model(self, new_data_num):
        min_valid_idx = max(0, self._total_data_idx - self._window_size - new_data_num)
        valid_num = self._newest_model_data_idx - min_valid_idx
        out_num = max(0, self._new_embeddings.shape[0] - valid_num)
        if valid_num <= 0:
            print("model update is too slow, newest model is out of data!")
            replace_data_num = self._new_embeddings.shape[0]
            unreplace_data_num = 0
        else:
            replace_data_num = valid_num
            unreplace_data_num = self.pre_embeddings.shape[0] - replace_data_num

        self.pre_embeddings[:replace_data_num] = self._new_embeddings[-replace_data_num:]

        self.isomap_list = self._new_isomap_list
        self.transformation_info_list = self._new_transformation_info_list
        self.pre_cluster_num = self._new_cluster_num
        self.global_embedding_mean = self._new_global_embedding_mean
        self._new_cluster_indices = np.array(self._new_cluster_indices)
        self.cluster_indices = self._new_cluster_indices
        tmp_cluster_indices = self._new_cluster_indices - out_num
        # 为模型更新期间到达的数据分配聚类
        if unreplace_data_num > 0:
            cur_cluster_idx = np.zeros(self.pre_cluster_num)
            arr_data_cluster = np.array(self._data_cluster)
            for i in range(self.pre_cluster_num):
                valid_indices = tmp_cluster_indices[i][tmp_cluster_indices[i] >= 0]
                cur_cluster_idx[i] = np.mean(arr_data_cluster[valid_indices])

            dists = np.abs(np.repeat(arr_data_cluster[replace_data_num:][:, np.newaxis], axis=1,
                                     repeats=self.pre_cluster_num) -
                           np.repeat(cur_cluster_idx[np.newaxis, :], axis=0, repeats=unreplace_data_num))
            corr_cluster_idx = np.argmin(dists, axis=1)
            for i in range(unreplace_data_num):
                c_id = corr_cluster_idx[i]
                self.cluster_indices[c_id] = np.append(self.cluster_indices[c_id], replace_data_num + i)
                # self.isomap_list[c_id].merge_new_data_info(new_data,
                #                                            self.pre_embeddings[replace_data_num + i][np.newaxis, :],
                #                                            self._geodesic_dists[replace_data_num +
                #                                                                 i - self.initial_train_num])
        return out_num

    def _get_new_model_info(self):
        self._new_embeddings, self._new_isomap_list, self._new_transformation_info_list, \
            self._new_global_embedding_mean, self._new_cluster_indices, self._newest_model_data_idx\
            = self._model_return_queue.get()
        self._new_cluster_num = len(self._new_cluster_indices)
        self._get_new_model = True


class SIsomapUpdater(Process, SIsomapPlus):
    def __init__(self, model_update_queue, model_return_queue, update_finish_queue, train_num, n_components, n_neighbors, epsilon=0.25):
        Process.__init__(self, name="SIsomapUpdater")
        SIsomapPlus.__init__(self, train_num, n_components, n_neighbors, epsilon)
        self._model_update_queue = model_update_queue
        self._model_return_queue = model_return_queue
        self._update_finish_queue = update_finish_queue

    def run(self) -> None:
        while True:
            stop_flag, data, cur_data_idx = self._model_update_queue.get()

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
                                          copy.copy(self.cluster_indices), cur_data_idx])
            self.isomap_list = []
            self.transformation_info_list = []
            self.global_embedding_mean = []
            self.cluster_indices = None
            self._update_finish_queue.put(True)
