import os
import time
from multiprocessing import Process

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist

from model.ine import INEModel
from model.scdr.scdr_parallel import SCDRParallel
from model.stream_isomap import SIsomapPlus
from utils.constant_pool import METRIC_NAMES, STEADY_METRIC_NAMES
from utils.queue_set import ModelUpdateQueueSet
from dataset.streaming_data_mock import StreamingDataMock, SimulatedStreamingData, StreamingDataMock2Stage
from model.scdr.dependencies.experiment import position_vis
from model.atSNE import atSNEModel
from model.si_pca import StreamingIPCA
from model.xtreaming import XtreamingModel
from utils.common_utils import evaluate_and_log, time_stamp_to_date_time_adjoin
from utils.logger import InfoLogger
from utils.metrics_tool import Metric, cal_global_position_change, cal_neighbor_pdist_change, cal_manifold_pdist_change
from utils.nn_utils import compute_knn_graph, get_pairwise_distance
from utils.queue_set import StreamDataQueueSet


plt.rcParams['animation.ffmpeg_path'] = r'H:\Softwares\ffmpeg\bin\ffmpeg.exe'


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
        self.save_embedding_iter = cfg.exp_params.save_iter
        # self._eval_nums = None
        self._eval_nums = 2
        self._eval_iter = cfg.exp_params.eval_iter
        self.log_path = log_path
        self.result_save_dir = result_save_dir
        self.model = None
        self.n_components = cfg.exp_params.latent_dim
        self.cur_time_step = 0
        self.do_eval = do_eval
        self.metric_tool = None
        self.eval_k = 10
        self.vc_k = self.eval_k

        self._key_time = 0
        self.pre_embedding = None
        self.cur_embedding = None
        self.embedding_dir = None
        self.img_dir = None

        self.initial_cached = False
        self.initial_data_num = cfg.exp_params.initial_data_num
        self.initial_data = None
        self.initial_labels = None
        self.history_data = None
        self.history_label = None

        self.cls2idx = {}
        self.debug = True
        self.save_img = True
        self._make_animation = cfg.exp_params.make_animation
        self.save_embedding_npy = False

        # 实验用
        self._x_min = 1e7
        self._x_max = 1e-7
        self._y_min = 1e7
        self._y_max = 1e-7
        self._embeddings_histories = []
        self._faith_metric_records = [[] for item in METRIC_NAMES]
        self._steady_metric_records = [[] for item in STEADY_METRIC_NAMES]

    def _prepare_streaming_data(self):
        # self.streaming_mock = StreamingDataMock(self.dataset_name, self.cfg.exp_params.stream_rate, self.seq_indices)
        self.streaming_mock = StreamingDataMock2Stage(self.dataset_name, None, 0,
                                                      self.cfg.exp_params.stream_rate, self.seq_indices)

    def _train_begin(self):
        self._prepare_streaming_data()
        self.result_save_dir = os.path.join(self.result_save_dir, self.dataset_name,
                                            time_stamp_to_date_time_adjoin(int(time.time())))
        check_path_exist(self.result_save_dir)
        self.log = open(os.path.join(self.result_save_dir, self.log_path), 'a')

        if self._eval_nums is not None:
            time_steps = self.streaming_mock.time_step_num
            self._eval_iter = time_steps // self._eval_nums

    def build_metric_tool(self):
        data = self.history_data
        targets = self.history_label
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

    def start_ine(self):
        self.model = INEModel(self.cfg.exp_params.initial_data_num, self.n_components,
                              self.cfg.method_params.n_neighbors)
        self.stream_fitting()

    def start_sisomap(self):
        self.model = SIsomapPlus(self.cfg.exp_params.initial_data_num, self.n_components,
                                 self.cfg.method_params.n_neighbors)
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
            stream_data, stream_labels, _ = self.streaming_mock.next_time_data()

            self._project_pipeline(stream_data, stream_labels)

    def _project_pipeline(self, stream_data, stream_labels):
        pre_labels = self.history_label
        cache_flag, stream_data, stream_labels = self._cache_initial(stream_data, stream_labels)
        if cache_flag:
            self.pre_embedding = self.cur_embedding
            if self.cur_time_step > 1 and isinstance(self.model, SCDRParallel):
                self.model.stream_dataset.add_new_data(data=stream_data)
            sta = time.time()
            ret_embeddings = self.model.fit_new_data(stream_data, stream_labels)
            if self.cur_time_step > 2:
                self._key_time += time.time() - sta

            if ret_embeddings is not None:
                cur_x_min, cur_y_min = np.min(ret_embeddings, axis=0)
                cur_x_max, cur_y_max = np.max(ret_embeddings, axis=0)
                self._x_min = min(self._x_min, cur_x_min)
                self._x_max = max(self._x_max, cur_x_max)
                self._y_min = min(self._y_min, cur_y_min)
                self._y_max = max(self._y_max, cur_y_max)
                self._embeddings_histories.append(ret_embeddings)
                self.evaluate(self.pre_embedding, ret_embeddings, pre_labels)
                self.cur_embedding = ret_embeddings
                self.save_embeddings_info(self.cur_embedding)

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

    def save_embeddings_info(self, cur_embeddings, custom_id=None, train_end=False):
        custom_id = self.cur_time_step if custom_id is None else custom_id
        if self.save_embedding_npy and (self.cur_time_step % self.save_embedding_iter == 0 or train_end):
            np.save(os.path.join(self.embedding_dir, "t_{}.npy".format(custom_id)), self.cur_embedding)

        if self.cur_time_step % self.vis_iter == 0 or train_end:
            img_save_path = os.path.join(self.img_dir, "t_{}.jpg".format(custom_id)) if self.save_img else None
            position_vis(self.history_label[:cur_embeddings.shape[0]], img_save_path, cur_embeddings, None)

    def evaluate(self, pre_embeddings, cur_embeddings, pre_labels, train_end=False):
        if self.cur_time_step % self._eval_iter > 0 and not train_end:
            return

        self.build_metric_tool()

        faithful_results = self.metric_tool.cal_all_metrics(self.eval_k, cur_embeddings, knn_k=self.eval_k)
        faith_metric_output = ""
        for i, item in enumerate(METRIC_NAMES):
            faith_metric_output += " %s: %.4f" % (item, faithful_results[i])
            self._faith_metric_records[i].append(faithful_results[i])

        InfoLogger.info(faith_metric_output)

        position_change = cal_global_position_change(cur_embeddings, pre_embeddings)
        pdist_change = cal_manifold_pdist_change(cur_embeddings, pre_embeddings, pre_labels)

        steady_results = [position_change, pdist_change]
        steady_metric_output = ""
        for i, item in enumerate(STEADY_METRIC_NAMES):
            steady_metric_output += " %s: %.4f" % (item, steady_results[i])
            self._steady_metric_records[i].append(steady_results[i])

        InfoLogger.info(steady_metric_output)

        if self.log is not None:
            self.log.write(faith_metric_output + "\n")
            self.log.write(steady_metric_output + "\n")

    def train_end(self):

        if isinstance(self.model, XtreamingModel) and not self.model.buffer_empty():
            self.cur_embedding = self.model.fit()
            self.save_embeddings_info(self.cur_embedding, train_end=True)

        if isinstance(self.model, SCDRParallel):
            self.model.save_model()

        ret = self.model.ending()
        if self.log is not None:
            self.log.write(ret + "\n")

        end_time = time.time()
        output = "Key Cost Time: %.4f" % self._key_time
        InfoLogger.info(output)
        self.log.write(output + "\n")
        if self.do_eval:
            self.build_metric_tool()

        self.evaluate(self.pre_embedding, self.cur_embedding, self.history_label[:self.pre_embedding.shape[0]], True)

        self._metric_conclusion()

    def _metric_conclusion(self):

        def _draw(y, save_name):
            plt.figure()
            plt.title(save_name)
            plt.plot(x, y)
            plt.ylabel(save_name)
            # plt.legend()
            plt.savefig(os.path.join(save_dir, "{}.jpg".format(save_name)))
            plt.show()

        faith_metric_output = ""
        save_dir = os.path.join(self.img_dir, "metrics")
        check_path_exist(save_dir)
        x = np.arange(len(self._faith_metric_records[0]))
        for i, item in enumerate(METRIC_NAMES):
            faith_metric_output += " Avg %s: %.4f" % (item, float(np.mean(self._faith_metric_records[i])))
            _draw(self._faith_metric_records[i], item)
        InfoLogger.info(faith_metric_output)

        steady_metric_output = ""
        for i, item in enumerate(STEADY_METRIC_NAMES):
            steady_metric_output += " Avg %s: %.4f" % (item, float(np.mean(self._steady_metric_records[i])))
            _draw(self._steady_metric_records[i], item)
        InfoLogger.info(steady_metric_output)

        if self.log is not None:
            self.log.write(faith_metric_output + "\n")
            self.log.write(steady_metric_output + "\n")

        if self._make_animation:
            self._make_embedding_video(save_dir)

    def _make_embedding_video(self, save_dir):
        self._embeddings_histories = np.array(self._embeddings_histories)

        def _loose(d_min, d_max, rate=0.1):
            scale = d_max - d_min
            d_max += np.abs(scale * rate)
            d_min -= np.abs(scale * rate)
            return d_min, d_max

        l_x_min, l_x_max = _loose(self._x_min, self._x_max)
        l_y_min, l_y_max = _loose(self._y_min, self._y_max)

        fig, ax = plt.subplots()

        # def init():
        #     ax.set(xlim=(l_x_min, l_x_max), ylim=(l_y_min, l_y_max))
        #     ax.set_aspect('equal')
        #     return ax
        if len(np.unique(self.history_label)) > 10:
            color_list = "tab20"
        else:
            color_list = "tab10"

        def update(idx):
            if idx % 100 == 0:
                print("frame", idx)

            cur_embeddings = self._embeddings_histories[idx]
            ax.cla()
            ax.set(xlim=(l_x_min, l_x_max), ylim=(l_y_min, l_y_max))
            ax.set_aspect('equal')
            # ax.axis('equal')  # 将xy轴隐藏
            ax.scatter(x=cur_embeddings[:, 0], y=cur_embeddings[:, 1],
                       c=self.streaming_mock.seq_label[:cur_embeddings.shape[0]], s=2, cmap=color_list)
            ax.set_title("Timestep: {}".format(int(idx)))

        ani = FuncAnimation(fig, update, frames=self._embeddings_histories.shape[0], interval=33, blit=False)
        ani.save(os.path.join(save_dir, "embedding.mp4"), writer='ffmpeg', dpi=300)


class StreamingExProcess(StreamingEx, Process):
    def __init__(self, cfg, seq_indices, result_save_dir, log_path="log_streaming_process.txt", do_eval=True):
        self.name = "数据处理进程"
        self.stream_data_queue_set = StreamDataQueueSet()
        self.cdr_update_queue_set = None
        self.update_num = 0

        Process.__init__(self, name=self.name)
        StreamingEx.__init__(self, cfg, seq_indices, result_save_dir, log_path, do_eval)

    def _prepare_streaming_data(self):
        self.streaming_mock = SimulatedStreamingData(self.dataset_name, self.cfg.exp_params.stream_rate,
                                                     self.stream_data_queue_set, self.seq_indices)
        self.streaming_mock.start()
        # super()._prepare_streaming_data()

    def start_parallel_scdr(self, model_update_queue_set, model_trainer):
        self.cdr_update_queue_set = model_update_queue_set
        self.model = SCDRParallel(self.cfg.method_params.n_neighbors, self.cfg.method_params.batch_size,
                                  model_update_queue_set, self.cfg.exp_params.initial_data_num,
                                  ckpt_path=self.cfg.method_params.ckpt_path, device=model_trainer.device)
        model_trainer.daemon = True
        model_trainer.start()
        self.stream_fitting()

    def processing(self):
        self.run()

    def _get_stream_data(self, accumulate=False):
        if not self.stream_data_queue_set.stop_flag_queue.empty():
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
            if self.cdr_update_queue_set.WAITING_UPDATED_DATA.value == 1:
                self._update_scdr_model()

            # 获取数据
            stream_end_flag, stream_data, stream_labels = self._get_stream_data(accumulate=False)
            if stream_end_flag:
                break

            self._project_pipeline(stream_data, stream_labels)

    def _update_scdr_model(self):
        embeddings, infer_model, stream_dataset, cluster_indices = self.cdr_update_queue_set.embedding_queue.get()
        total_embeddings, replace_model = self.model.update_scdr(infer_model, embeddings, stream_dataset)
        # TODO: 对于模型更新期间接收到的数据，如何对其进行处理以保证其余模型更新后的嵌入结果保持一致

        self.pre_embedding = self.cur_embedding
        if replace_model:
            self.save_embeddings_info(total_embeddings, custom_id="u{}".format(self.update_num), )
        self.cur_embedding = total_embeddings
        self.update_num += 1
        self.cdr_update_queue_set.WAITING_UPDATED_DATA.value = 0

    def train_end(self):
        # 结束流数据产生进程
        self.streaming_mock.stop_flag = True
        self.streaming_mock.kill()
        self.stream_data_queue_set.clear()

        # 结束流数据处理
        self.cur_embedding = self.model.get_final_embeddings()
        self.save_embeddings_info(self.cur_embedding, train_end=True)
        self.model.ending()

        # 结束模型更新进程
        self.cdr_update_queue_set.flag_queue.put(ModelUpdateQueueSet.STOP)

        super().train_end()
