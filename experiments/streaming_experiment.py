import os
import time

import numpy as np

from dataset.streaming_data_mock import StreamingDataMock
from experiments.experiment import position_vis
from model.atSNE import atSNEModel
from model.scdr import SCDRModel
from model.si_pca import StreamingIPCA
from model.xtreaming import XtreamingModel
from utils.common_utils import evaluate_and_log
from utils.logger import InfoLogger
from utils.metrics_tool import Metric
from utils.nn_utils import compute_knn_graph, get_pairwise_distance
from utils.time_utils import time_stamp_to_date_time_adjoin


def check_path_exist(t_path):
    if not os.path.exists(t_path):
        os.makedirs(t_path)


class StreamingEx:
    def __init__(self, initial_cls, cfg, seq_indices, result_save_dir,
                 log_path="log_streaming_con.txt", do_eval=True):
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
        self.sta_time = 0
        self.other_time = 0
        self.cur_embedding = None
        self.embedding_dir = None
        self.img_dir = None

        self.initial_cached = False
        self.initial_data_num = cfg.exp_params.initial_data_num
        self.initial_data = None
        self.initial_labels = None
        self.first_fit = False

        self.cls2idx = {}

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

    def start_siPCA(self, forgetting_factor):
        self.model = StreamingIPCA(self.n_components, forgetting_factor)
        self.stream_fitting()

    def start_atSNE(self, perplexity, finetune_iter, n_iter):
        self.model = atSNEModel(finetune_iter, n_components=self.n_components, perplexity=perplexity,
                                init="pca", n_iter=n_iter, verbose=0)
        self.stream_fitting()

    def start_xtreaming(self, buffer_size, eta):
        self.model = XtreamingModel(buffer_size, eta)
        self.stream_fitting()

    def start_scdr(self, n_neighbors, buffer_size, model_trainer, initial_train_epoch=400, finetune_epoch=50,
                   ckpt_path=None):
        self.model = SCDRModel(n_neighbors, buffer_size, model_trainer, initial_train_epoch, finetune_epoch,
                               ckpt_path=ckpt_path)
        self.stream_fitting()

    def stream_fitting(self):
        self._train_begin()
        if isinstance(self.model, SCDRModel):
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

            if not self.first_fit and isinstance(self.model, atSNEModel):
                self.first_fit = True
                ret_embeddings = self.model.fit_transform(stream_data, stream_labels)
            else:
                ret_embeddings = self.model.fit_new_data(stream_data, stream_labels)

            if ret_embeddings is not None:
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

    def save_embeddings_imgs(self):
        sta = time.time()
        np.save(os.path.join(self.embedding_dir, "t_{}.npy".format(self.cur_time_step)), self.cur_embedding)
        if self.cur_time_step % self.vis_iter == 0:
            position_vis(self.streaming_mock.seq_label,
                         os.path.join(self.img_dir, "t_{}.jpg".format(self.cur_time_step)),
                         self.cur_embedding)
        self.other_time += time.time() - sta

    def train_end(self):
        if isinstance(self.model, XtreamingModel) and not self.model.buffer_empty():
            self.cur_embedding = self.model.fit()
            self.save_embeddings_imgs()
        elif isinstance(self.model, SCDRModel) and not self.model.buffer_empty():
            self.cur_embedding = self.model.clear_buffer(final_embed=True)
            self.save_embeddings_imgs()

        if isinstance(self.model, SCDRModel):
            self.model.save_model()

        end_time = time.time()
        output = "Total Cost Time: %.4f" % (end_time - self.sta_time - self.other_time)
        InfoLogger.info(output)
        self.log.write(output + "\n")
        if self.do_eval:
            self.build_metric_tool()
        evaluate_and_log(self.metric_tool, self.cur_embedding, self.eval_k, knn_k=10, log_file=self.log)
