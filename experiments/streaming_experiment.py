import os
import time

import numpy as np
from scipy.spatial.distance import cdist

from dataset.streaming_data_mock import StreamingDataMock
from dataset.warppers import eval_knn_acc
from experiments.experiment import position_vis
from model.atSNE import atSNEModel
from model.scdr import SCDRModel, SCDRBase
from model.scdr_rt import RTSCDRModel
from model.si_pca import StreamingIPCA
from model.xtreaming import XtreamingModel
from utils.common_utils import evaluate_and_log
from utils.logger import InfoLogger
from utils.metrics_tool import Metric, metric_mental_map_preservation, metric_mental_map_preservation_cntp, \
    metric_mental_map_preservation_edge, metric_visual_consistency_dbscan
from utils.nn_utils import compute_knn_graph, get_pairwise_distance
from utils.time_utils import time_stamp_to_date_time_adjoin


def check_path_exist(t_path):
    if not os.path.exists(t_path):
        os.makedirs(t_path)


def cal_kNN_change(new_data, new_knn_indices, new_knn_dists, pre_data, pre_knn_indices, pre_knn_dists):
    dists = cdist(new_data, pre_data)
    pre_n_samples = pre_data.shape[0]
    new_n_samples = new_data.shape[0]
    neighbor_changed_indices = set(np.arange(0, new_n_samples, 1) + pre_n_samples)
    pre_farest_neighbor_dist = np.max(pre_knn_dists, axis=1)

    for i in range(new_n_samples):
        indices = np.where(dists[i] - pre_farest_neighbor_dist < 0)[0]
        if len(indices) <= 0:
            continue
        for j in indices:
            if j not in neighbor_changed_indices:
                neighbor_changed_indices.add(j)
            t = np.argmax(pre_knn_dists[j])
            if pre_knn_indices[j][t] not in neighbor_changed_indices:
                neighbor_changed_indices.add(pre_knn_indices[j][t])

            # 这个更新的过程应该是迭代的
            pre_knn_dists[j][t] = dists[i][j]
            pre_farest_neighbor_dist[j] = np.max(pre_knn_dists[j])
            pre_knn_indices[j][t] = pre_n_samples + i
    total_knn_indices = np.concatenate([pre_knn_indices, new_knn_indices], axis=0)
    total_knn_dists = np.concatenate([pre_knn_dists, new_knn_dists], axis=0)
    return list(neighbor_changed_indices), total_knn_indices, total_knn_dists


class StreamingEx:
    def __init__(self, initial_cls, cfg, seq_indices, result_save_dir, log_path="log_streaming_con.txt", do_eval=True):
        self.cfg = cfg
        self.dataset_name = cfg.exp_params.dataset
        # self.streaming_mock = StreamingDataMock2Stage(self.dataset_name, initial_cls, 1,
        #                                               args.exp_params.stream_rate, seq_indices=seq_indices)
        self.streaming_mock = StreamingDataMock(self.dataset_name, cfg.exp_params.stream_rate, seq_indices)
        self.vis_iter = cfg.exp_params.vis_iter
        self.log_path = log_path
        self.result_save_dir = result_save_dir
        self.model = None
        self.n_components = 2
        self.cur_time_step = 0
        self.do_eval = do_eval
        self.metric_tool = None
        self.eval_k = 7
        self.vc_k = 30
        self.pre_high_knn_indices = None
        self.pre_high_knn_dists = None

        self.sta_time = 0
        self.other_time = 0
        self.pre_embedding = None
        self.cur_embedding = None
        self.embedding_dir = None
        self.img_dir = None

        self.initial_cached = False
        self.initial_data_num = cfg.exp_params.initial_data_num
        self.initial_data = None
        self.initial_labels = None
        self.first_fit = False
        self.cur_data = None

        self.cls2idx = {}
        self.debug = True

    def _train_begin(self):
        self.sta_time = time.time()
        self.result_save_dir = os.path.join(self.result_save_dir, self.dataset_name,
                                            time_stamp_to_date_time_adjoin(int(time.time())))
        check_path_exist(self.result_save_dir)
        self.log = open(os.path.join(self.result_save_dir, self.log_path), 'a')

    def build_metric_tool(self):
        data = self.streaming_mock.history_data
        targets = self.streaming_mock.history_label
        knn_indices, knn_dists = compute_knn_graph(data, None, self.eval_k, None, accelerate=False)
        pairwise_distance = get_pairwise_distance(data, pairwise_distance_cache_path=None, preload=False)
        self.metric_tool = Metric(self.dataset_name, data, targets, knn_indices, knn_dists, pairwise_distance,
                                  k=self.eval_k)

    def start_siPCA(self):
        self.model = StreamingIPCA(self.n_components, self.cfg.method_params.forgetting_factor)
        self.stream_fitting()

    def start_atSNE(self):
        self.model = atSNEModel(self.cfg.method_params.finetune_iter, n_components=self.n_components,
                                perplexity=self.cfg.method_params.perplexity,
                                init="pca", n_iter=self.cfg.method_params.initial_train_iter, verbose=0)
        self.stream_fitting()

    def start_xtreaming(self):
        self.model = XtreamingModel(self.cfg.method_params.buffer_size, self.cfg.method_params.eta)
        self.stream_fitting()

    def start_scdr(self, model_trainer):
        self.model = SCDRModel(self.cfg.method_params.n_neighbors, self.cfg.method_params.buffer_size, model_trainer,
                               self.cfg.exp_params.initial_data_num,  self.cfg.method_params.initial_train_epoch,
                               self.cfg.method_params.finetune_epoch, ckpt_path=self.cfg.method_params.ckpt_path)
        self.stream_fitting()

    def start_rtscdr(self, model_trainer):
        self.model = RTSCDRModel(self.cfg.method_params.shift_buffer_size, self.cfg.method_params.n_neighbors,
                                 model_trainer,  self.cfg.exp_params.initial_data_num,
                                 self.cfg.method_params.initial_train_epoch, self.cfg.method_params.finetune_epoch,
                                 ckpt_path=self.cfg.method_params.ckpt_path)

        self.stream_fitting()

    def stream_fitting(self):
        self._train_begin()
        if isinstance(self.model, SCDRBase):
            self.model.model_trainer.result_save_dir = self.result_save_dir

        # initial_data, initial_label = self.streaming_mock.next_time_data()
        # if isinstance(self.model, atSNEModel):
        #     ret_embeddings = self.model.fit_transform(initial_data, initial_label)
        # else:
        #     ret_embeddings = self.model.fit_new_data(initial_data, initial_label)
        #
        # if ret_embeddings is not None:
        #     self.cur_embedding = ret_embeddings
        #     initial_save_path = os.path.join(self.result_save_dir, "t_{}.jpg".format(self.cur_time_step))
        #     position_vis(initial_label, initial_save_path, self.cur_embedding)

        self.processing()
        self.train_end()

    def processing(self):
        self.embedding_dir = os.path.join(self.result_save_dir, "embeddings")
        check_path_exist(self.embedding_dir)
        self.img_dir = os.path.join(self.result_save_dir, "imgs")
        check_path_exist(self.img_dir)
        for i in range(self.streaming_mock.time_step_num):
            self.cur_time_step += 1
            output = "Start processing timestamp {}".format(i + 1)
            InfoLogger.info(output)
            stream_data, stream_labels = self.streaming_mock.next_time_data()

            if not self.initial_cached:
                cached = self._cache_initial_data(stream_data, stream_labels)
                if cached:
                    stream_data = self.initial_data
                    stream_labels = self.initial_labels
                else:
                    continue
            self.cur_data = stream_data
            if not self.first_fit and isinstance(self.model, atSNEModel):
                self.first_fit = True
                ret_embeddings = self.model.fit_transform(stream_data, stream_labels)
            else:
                ret_embeddings = self.model.fit_new_data(stream_data, stream_labels)

            if ret_embeddings is not None:
                self.pre_embedding = self.cur_embedding
                self.cur_embedding = ret_embeddings
                self.save_embeddings_imgs()

    def _cache_initial_data(self, data, label=None):
        if self.initial_data is None:
            self.initial_data = data
            if label is not None:
                self.initial_labels = label
        else:
            self.initial_data = np.concatenate([self.initial_data, data], axis=0)
            if label is not None:
                self.initial_labels = np.concatenate([self.initial_labels, label])

        if self.initial_data.shape[0] >= self.initial_data_num:
            self.initial_cached = True
            return True
        else:
            return False

    def save_embeddings_imgs(self, force=False):
        sta = time.time()
        np.save(os.path.join(self.embedding_dir, "t_{}.npy".format(self.cur_time_step)), self.cur_embedding)
        if self.cur_time_step % self.vis_iter == 0 or force:
            title = None
            if self.pre_embedding is not None:
                pre_embeddings = self.pre_embedding
                cur_embeddings = self.cur_embedding

                # pre_embeddings = self.pre_embedding - np.expand_dims(np.mean(self.pre_embedding, axis=0), axis=0)
                # cur_embeddings = self.cur_embedding - np.expand_dims(np.mean(self.cur_embedding, axis=0), axis=0)
                # 投影函数没变，但是增加了新的点之后，mean会有多大变化
                # print("Previous mean:", np.mean(self.pre_embedding, axis=0))
                # print("Current mean:", np.mean(self.cur_embedding, axis=0))

                cur_nn_indices, cur_nn_dists = compute_knn_graph(cur_embeddings, None, self.vc_k, None)
                pre_nn_indices, pre_nn_dists = compute_knn_graph(pre_embeddings, None, self.vc_k, None)

                # vc_point = metric_mental_map_preservation(cur_embeddings, pre_embeddings, cur_nn_indices, self.vc_k)[0]
                # vc_edge = metric_mental_map_preservation_edge(cur_embeddings, pre_embeddings, cur_nn_indices)[0]

                # vc_cntp = metric_mental_map_preservation_cntp(self.cur_embedding, self.pre_embedding)
                vc_cntp = 0
                high_dist2new_data = cdist(self.cur_data, self.streaming_mock.history_data) ** 2

                # cur_high_nn_indices, cur_high_nn_dists = compute_knn_graph(self.streaming_mock.history_data, None,
                #                                                            self.vc_k, None)
                # new_n_samples = cur_embeddings.shape[0] - pre_embeddings.shape[0]
                # new_nn_indices = cur_high_nn_indices[-new_n_samples:]
                # new_nn_dists = cur_high_nn_dists[-new_n_samples:]
                # new_data = self.streaming_mock.history_data[-new_n_samples:]
                # pre_data = self.streaming_mock.history_data[:-new_n_samples]
                # kNN_change_indices, self.pre_high_knn_indices, self.pre_high_knn_dists = \
                #     cal_kNN_change(new_data, new_nn_indices, new_nn_dists, pre_data, self.pre_high_knn_indices,
                #                    self.pre_high_knn_dists)
                # acc_list, a_acc_list = eval_knn_acc(cur_high_nn_indices, self.pre_high_knn_indices, new_n_samples, pre_embeddings.shape[0])

                vc_db = metric_visual_consistency_dbscan(cur_embeddings, pre_embeddings, np.mean(pre_nn_dists),
                                                         np.mean(cur_nn_dists), high_dist2new_data=high_dist2new_data)

                # title = "P: %.2f E: %.2f C: %.2f D: %.4f" % (vc_point, vc_edge, vc_cntp, vc_db)
                # print("VC Point = %.2f VC Edge = %.2f VC Cluster = %.2f VC DB = %.4f" % (
                # vc_point, vc_edge, vc_cntp, vc_db))
                title = "Visual Consistency: %.4f" % vc_db
                # print(title)
            else:
                self.pre_high_knn_indices, self.pre_high_knn_dists = compute_knn_graph(self.streaming_mock.history_data,
                                                                                       None, self.vc_k, None)

            position_vis(self.streaming_mock.seq_label,
                         os.path.join(self.img_dir, "t_{}.jpg".format(self.cur_time_step)),
                         self.cur_embedding, title)

        self.other_time += time.time() - sta

    def train_end(self):
        if isinstance(self.model, XtreamingModel) and not self.model.buffer_empty():
            self.cur_embedding = self.model.fit()
            self.save_embeddings_imgs(force=True)
        elif (isinstance(self.model, SCDRModel) or isinstance(self.model, RTSCDRModel)) and not self.model.buffer_empty():
            self.cur_embedding = self.model.clear_buffer(final_embed=True)
            self.save_embeddings_imgs(force=True)

        if isinstance(self.model, SCDRModel) or isinstance(self.model, RTSCDRModel):
            self.model.save_model()
            if self.debug:
                self.model.print_time_cost_info()

        end_time = time.time()
        output = "Total Cost Time: %.4f" % (end_time - self.sta_time - self.other_time)
        InfoLogger.info(output)
        self.log.write(output + "\n")
        if self.do_eval:
            self.build_metric_tool()
        evaluate_and_log(self.metric_tool, self.cur_embedding, self.eval_k, knn_k=10, log_file=self.log)

