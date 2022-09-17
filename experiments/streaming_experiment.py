import os
import time
from multiprocessing import Process

import numpy as np
from scipy.spatial.distance import cdist

from model.scdr.scdr_parallel import SCDRParallel
from utils.queue_set import ModelUpdateQueueSet
from dataset.streaming_data_mock import StreamingDataMock, SimulatedStreamingData
from model.scdr.dependencies.experiment import position_vis
from model.atSNE import atSNEModel
from model.si_pca import StreamingIPCA
from model.xtreaming import XtreamingModel
from utils.common_utils import evaluate_and_log, time_stamp_to_date_time_adjoin
from utils.logger import InfoLogger
from utils.metrics_tool import Metric, metric_visual_consistency_simplest
from utils.nn_utils import compute_knn_graph, get_pairwise_distance
from utils.queue_set import StreamDataQueueSet


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
    def __init__(self, cfg, seq_indices, result_save_dir, log_path="log_streaming_con.txt", do_eval=True):
        self.cfg = cfg
        self.seq_indices = seq_indices
        self.dataset_name = cfg.exp_params.dataset
        self.streaming_mock = None
        self.vis_iter = cfg.exp_params.vis_iter
        self.log_path = log_path
        self.result_save_dir = result_save_dir
        self.model = None
        self.n_components = 2
        self.cur_time_step = 0
        self.do_eval = do_eval
        self.metric_tool = None
        self.fixed_k = 10
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
        self.history_data = None
        self.history_label = None

        self.cls2idx = {}
        self.debug = True
        self.save_img = False
        self.save_embedding_npy = False
        self.metric_vs = False

    def _prepare_streaming_data(self):
        self.streaming_mock = StreamingDataMock(self.dataset_name, self.cfg.exp_params.stream_rate, self.seq_indices)

    def _train_begin(self):
        self._prepare_streaming_data()
        self.sta_time = time.time()
        self.result_save_dir = os.path.join(self.result_save_dir, self.dataset_name,
                                            time_stamp_to_date_time_adjoin(int(time.time())))
        check_path_exist(self.result_save_dir)
        self.log = open(os.path.join(self.result_save_dir, self.log_path), 'a')

    def build_metric_tool(self):
        data = self.history_data
        targets = self.history_label
        knn_indices, knn_dists = compute_knn_graph(data, None, self.fixed_k, None, accelerate=False)
        pairwise_distance = get_pairwise_distance(data, pairwise_distance_cache_path=None, preload=False)
        self.metric_tool = Metric(self.dataset_name, data, targets, knn_indices, knn_dists, pairwise_distance,
                                  k=self.fixed_k)

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

    def stream_fitting(self):
        self._train_begin()

        self.processing()
        self.train_end()

    def _cache_initial(self, stream_data, stream_labels):
        self.history_data = stream_data if self.history_data is None else np.concatenate(
            [self.history_data, stream_data], axis=0)
        if stream_labels is not None:
            self.history_label = stream_labels if self.history_label is None else np.concatenate(
                [self.history_label, stream_labels])

        if self.initial_cached:
            return True, stream_data, stream_labels
        else:
            cached = self._cache_initial_data(stream_data, stream_labels)
            if cached:
                stream_data = self.initial_data
                stream_labels = self.initial_labels
            return cached, stream_data, stream_labels

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

            self._project_pipeline(stream_data, stream_labels)

    def _project_pipeline(self, stream_data, stream_labels):
        cache_flag, stream_data, stream_labels = self._cache_initial(stream_data, stream_labels)
        if cache_flag:
            self.cur_data = stream_data
            ret_embeddings = self.model.fit_new_data(stream_data, stream_labels)

            if ret_embeddings is not None:
                self.pre_embedding = self.cur_embedding
                self.cur_embedding = ret_embeddings
                self.save_embeddings_imgs(self.pre_embedding, self.cur_embedding)

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

    def save_embeddings_imgs(self, pre_embeddings, cur_embeddings, force_vc=False, custom_id=None):
        sta = time.time()
        if self.save_embedding_npy:
            custom_id = self.cur_time_step if custom_id is None else custom_id
            np.save(os.path.join(self.embedding_dir, "t_{}.npy".format(custom_id)), self.cur_embedding)

        # cur_low_nn_indices = compute_knn_graph(self.cur_embedding, None, self.vc_k, None)[0]
        # cur_high_nn_indices = compute_knn_graph(self.streaming_mock.history_data, None, self.vc_k, None)[0]
        # preserve, fake_intro = metric_neighbor_preserve_introduce(cur_low_nn_indices, cur_high_nn_indices)
        # print("Preserve Rate: %.4f Fake Intro Rate: %.4f" % (preserve, fake_intro))

        if self.cur_time_step % self.vis_iter == 0 or force_vc:
            title = None
            if self.pre_embedding is not None:
                # ===============================================考虑新数据对VC的影响============================================
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
                high_dist2new_data = cdist(self.cur_data, self.history_data) ** 2

                # cur_high_nn_indices, cur_high_nn_dists = compute_knn_graph(self.streaming_mock.history_data, None,
                # self.vc_k, None)
                # new_n_samples = cur_embeddings.shape[0] - pre_embeddings.shape[0]
                # new_nn_indices = cur_high_nn_indices[-new_n_samples:]
                # new_nn_dists = cur_high_nn_dists[-new_n_samples:]
                # new_data = self.streaming_mock.history_data[-new_n_samples:]
                # pre_data = self.streaming_mock.history_data[:-new_n_samples]
                # kNN_change_indices, self.pre_high_knn_indices, self.pre_high_knn_dists = \
                #     cal_kNN_change(new_data, new_nn_indices, new_nn_dists, pre_data, self.pre_high_knn_indices,
                #                    self.pre_high_knn_dists)
                # acc_list, a_acc_list = eval_knn_acc(cur_high_nn_indices, self.pre_high_knn_indices, new_n_samples, pre_embeddings.shape[0])

                # vc_inconsistency = metric_visual_consistency_dbscan(cur_embeddings, pre_embeddings,
                #                                                     np.mean(pre_nn_dists), np.mean(cur_nn_dists),
                #                                                     high_dist2new_data=high_dist2new_data)

                # title = "P: %.2f E: %.2f C: %.2f D: %.4f" % (vc_point, vc_edge, vc_cntp, vc_db)
                # print("VC Point = %.2f VC Edge = %.2f VC Cluster = %.2f VC DB = %.4f" % (
                # vc_point, vc_edge, vc_cntp, vc_db))
                # =======================================================================================================================

                # =========================================不考虑新数据的影响================================================
                vc_inconsistency = metric_visual_consistency_simplest(cur_embeddings, pre_embeddings) if self.metric_vs else 0
                # ========================================================================================================

                title = "Visual Inconsistency: %.4f" % vc_inconsistency
                # print(title)
            else:
                self.pre_high_knn_indices, self.pre_high_knn_dists = compute_knn_graph(self.history_data,
                                                                                       None, self.vc_k, None)

            img_save_path = os.path.join(self.img_dir, "t_{}.jpg".format(custom_id)) if self.save_img else None
            position_vis(self.streaming_mock.seq_label[:cur_embeddings.shape[0]], img_save_path, cur_embeddings, title)

        self.other_time += time.time() - sta

    def train_end(self):

        if isinstance(self.model, XtreamingModel) and not self.model.buffer_empty():
            self.cur_embedding = self.model.fit()
            self.save_embeddings_imgs(self.pre_embedding, self.cur_embedding, force_vc=True)

        if isinstance(self.model, SCDRParallel):
            self.model.save_model()
            if self.debug:
                self.model.ending()

        end_time = time.time()
        output = "Total Cost Time: %.4f" % (end_time - self.sta_time - self.other_time)
        InfoLogger.info(output)
        self.log.write(output + "\n")
        if self.do_eval:
            self.build_metric_tool()
        evaluate_and_log(self.metric_tool, self.cur_embedding, self.fixed_k, knn_k=self.fixed_k, log_file=self.log)
        self.model.ending()


class StreamingExProcess(StreamingEx, Process):
    def __init__(self, cfg, seq_indices, result_save_dir, log_path="log_streaming_process.txt", do_eval=True):
        self.name = "数据处理进程"
        self.stream_end_flag = False
        self.stream_data_queue_set = StreamDataQueueSet()
        self.cdr_update_queue_set = None
        self.update_num = 0

        Process.__init__(self, name=self.name)
        StreamingEx.__init__(self, cfg, seq_indices, result_save_dir, log_path, do_eval)

    def _prepare_streaming_data(self):
        self.streaming_mock = SimulatedStreamingData(self.dataset_name, self.cfg.exp_params.stream_rate,
                                                     self.stream_data_queue_set,
                                                     self.seq_indices)
        # self.streaming_mock = RealStreamingData(self.dataset_name, self.queue_set)
        self.streaming_mock.start()

    def start_parallel_scdr(self, model_update_queue_set, data_process_queue_set, model_trainer, data_processor):
        self.cdr_update_queue_set = model_update_queue_set
        self.model = SCDRParallel(model_update_queue_set, data_process_queue_set, self.cfg.exp_params.initial_data_num,
                                  self.cfg.method_params.initial_train_epoch, ckpt_path=self.cfg.method_params.ckpt_path,
                                  device=model_trainer.device)
        model_trainer.daemon = True
        model_trainer.start()
        data_processor.daemon = True
        data_processor.start()
        self.stream_fitting()
        model_update_queue_set.STOP.value = 1
        data_process_queue_set.STOP.value = 1

    def processing(self):
        self.run()

    def _get_stream_data(self, accumulate=False):
        if not self.stream_end_flag and not self.stream_data_queue_set.stop_flag_queue.empty():
            self.stream_end_flag = self.stream_data_queue_set.stop_flag_queue.get()

        if self.stream_end_flag and self.stream_data_queue_set.data_queue.empty():
            return True, None, None

        data_and_labels = self._get_data_accumulate() if accumulate else self._get_data_single()
        total_data, total_labels = [], []
        for item in data_and_labels:
            total_data.append(item[0])
            if item[1] is not None:
                total_labels.append(item[1])
        total_data = np.array(total_data, dtype=float)
        total_labels = None if len(total_labels) == 0 else np.array(total_labels, dtype=int)

        output = "Start processing timestamp {}".format(self.cur_time_step)
        InfoLogger.info(output)

        return False, total_data, total_labels

    def _get_data_accumulate(self):
        total_data = self.stream_data_queue_set.data_queue.get()
        acc_num = 1
        while not self.stream_data_queue_set.data_queue.empty():
            acc_num += 1
            total_data.extend(self.stream_data_queue_set.data_queue.get())
        self.cur_time_step += acc_num
        return total_data

    def _get_data_single(self):
        data = self.stream_data_queue_set.data_queue.get()
        self.cur_time_step += 1
        return data

    def run(self) -> None:
        self.embedding_dir = os.path.join(self.result_save_dir, "embeddings")
        check_path_exist(self.embedding_dir)
        self.img_dir = os.path.join(self.result_save_dir, "imgs")
        check_path_exist(self.img_dir)
        # 开始产生数据
        self.stream_data_queue_set.start_flag_queue.put(True)
        while True:
            if isinstance(self.model, SCDRParallel):
                if (self.update_num == 0 and self.cdr_update_queue_set.INITIALIZING.value) \
                        or not self.cdr_update_queue_set.embedding_queue.empty():

                    if self.update_num == 0:
                        # 第一次训练模型的时候，接受数据的进程在这里被阻塞住了。合理的，实际上pre-existing data用来训练，这个过程不应该被记入流数据处理过程。
                        ret_embeddings, infer_model = self.cdr_update_queue_set.embedding_queue.get()
                        embeddings_when_update = None
                        self.model.update_scdr(infer_model, True, ret_embeddings, ret_embeddings.shape[0])
                        self.cdr_update_queue_set.INITIALIZING.value = False
                    else:
                        data_num_when_update, infer_model = self.cdr_update_queue_set.embedding_queue.get()
                        self.model.update_scdr(infer_model, embedding_num=data_num_when_update)
                        ret_embeddings = self.model.embed_current_data()
                        embeddings_when_update = self.pre_embedding[:data_num_when_update]

                    self.pre_embedding = self.cur_embedding
                    self.save_embeddings_imgs(embeddings_when_update, ret_embeddings,
                                              custom_id="u{}".format(self.update_num), force_vc=True)
                    self.cur_embedding = ret_embeddings
                    self.update_num += 1

            # 获取数据
            stream_end_flag, stream_data, stream_labels = self._get_stream_data(accumulate=False)
            if stream_end_flag:
                self.cdr_update_queue_set.flag_queue.put(ModelUpdateQueueSet.DATA_STREAM_END)
                break

            self._project_pipeline(stream_data, stream_labels)

    def train_end(self):
        self.streaming_mock.stop_flag = True
        self.streaming_mock.kill()
        self.stream_data_queue_set.clear()

        if isinstance(self.model, SCDRParallel):
            self.cur_embedding = self.model.model_update_final()
            self.save_embeddings_imgs(self.pre_embedding, self.cur_embedding, force_vc=True)

        super().train_end()

