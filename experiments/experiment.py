#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from model.dr_models.ModelSets import *
import pandas as pd
import math
from utils.metrics_tool import MetricProcess
from utils.math_utils import *
import matplotlib.pyplot as plt
import os
from utils.nn_utils import compute_knn_graph, get_pairwise_distance
from utils.queue_set import QueueSet
from utils.time_utils import *
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import shutil
from utils.constant_pool import ConfigInfo, METRIC_NAMES
from multiprocessing import Queue
from utils.logger import InfoLogger, LogWriter
import seaborn as sns


def draw_loss(training_loss, test_loss, idx, save_path=None):
    # 画出总损失图
    plt.figure()
    plt.plot(idx, training_loss, color="blue", label="training loss")
    if len(test_loss) > 0:
        plt.plot(idx, test_loss, color="red", label="test loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def position_vis(c, vis_save_path, z):
    x = z[:, 0]
    y = z[:, 1]
    c = np.array(c, dtype=int)

    plt.figure(figsize=(8, 8))
    if c is None:
        sns.scatterplot(x=x, y=y, s=1, legend=False, alpha=0.9)
    else:
        classes = np.unique(c)
        num_classes = classes.shape[0]
        palette = "tab10" if num_classes <= 10 else "tab20"
        sns.scatterplot(x=x, y=y, hue=c, s=8, palette=palette, legend=False, alpha=1.0)

    # plt.title(title, fontsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.axis("equal")

    # plt.title("{} Embeddings".format(method_name), fontsize=20)
    if vis_save_path is not None:
        plt.savefig(vis_save_path, dpi=800, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def results_to_excel(metric_names, metric_num, results, time_costs, save_path):
    mean_results = np.mean(results, axis=0)
    mean_time = np.mean(time_costs)
    time_costs.append(mean_time)
    results = np.concatenate([results, np.expand_dims(mean_results, 0)], axis=0)
    dic_res = {}
    for i in range(metric_num):
        dic_res[metric_names[i]] = results[:, i]
    dic_res["Time"] = time_costs
    df = pd.DataFrame(dic_res)
    df.to_excel(save_path)
    return mean_results, mean_time


def draw_projections(z, c, vis_save_path):
    position_vis(c, vis_save_path, z)


class Experiment:
    def __init__(self, model, dataset_name, configs, result_save_dir, config_path, shuffle, device='cuda',
                 log_path="logs.txt", multi=False):
        self.model = model
        self.config_path = config_path
        self.configs = configs
        self.device = device
        self.result_save_dir_modified = False
        self.result_save_dir = result_save_dir
        self.dataset_name = dataset_name
        self.train_loader = None

        self.n_samples = 0
        self.n_neighbors = configs.method_params.n_neighbors
        self.batch_num = 0

        self.vis_inter = 0
        self.epoch_num = configs.method_params.initial_train_epoch
        self.start_epoch = 0
        self.batch_size = configs.method_params.batch_size
        self.lr = configs.method_params.LR
        self.ckp_save_dir = None
        self.test_inter = int(configs.debug_params.eval_inter * self.epoch_num)
        self.print_iter = int(configs.debug_params.epoch_print_inter * self.epoch_num)
        self.launch_date_time = None
        self.optimizer = None
        self.scheduler = None

        # self.knn_indices = None
        # self.knn_dists = None
        # self.pairwise_distance = None
        # self.eval_knn_indices = None
        # self.eval_knn_dists = None
        # self.eval_pairwise_distance = None

        self.shuffle = shuffle
        self.is_image = isinstance(configs.exp_params.input_dims, list)
        self.normalize_method = "umap"
        self.eval_time_cost = 0
        self.total_time_cost = 0
        self.train_time_cost = 0
        self.multi = multi
        self.device_id = self.configs.local_rank if self.multi else int(self.device.split(":")[1])

        # 日志模块记录
        self.tmp_log_path = log_path
        self.log_process = None
        self.log_path = None
        self.message_queue = Queue()

        # 定量测量
        self.metric_tool = None
        self.val_metric_tool = None
        self.queue_set = QueueSet()

        # 是否在子集上进行评估
        # self.sub_eval_ratio = self.configs.training_params.sub_eval_ratio
        # self.sub_test_eval_ratio = self.configs.training_params.sub_test_eval_ratio
        # self.sub_eval_indices = None
        # self.sub_test_eval_indices = None

        # self.noise_test = "noise" in self.dataset_name

        # self.symmetry_knn_indices = None
        # self.neighbor_row = []
        # self.neighbor_col = []
        # self.neighbor_nums = []

        self.pre_embeddings = None

        self.do_test = True
        self.do_vis = True
        self.save_model = True
        self.save_final_embeddings = True
        self.draw_loss = True
        self.print_time_info = True

        self.fixed_k = 10

    def _train_begin(self, launch_time_stamp=None):
        self.model = self.model.to(self.device)
        self.sta_time = time.time() if launch_time_stamp is None else launch_time_stamp
        if not self.multi or self.device_id == 0:
            InfoLogger.info("Start Training for {} Epochs".format(self.epoch_num))

            param_template = "Experiment Configurations: \nMethods: %s Dataset: %s Epochs: %d Batch Size: %d \n" \
                             "Learning rate: %4f Optimizer: %s\n"

            param_str = param_template % (
                self.configs.method_params.method, self.dataset_name, self.epoch_num, self.batch_size,
                self.lr, self.configs.method_params.optimizer)

            InfoLogger.info(param_str)
            self.message_queue.put(param_str)

        if self.launch_date_time is None:
            if launch_time_stamp is None:
                launch_time_stamp = int(time.time())
            self.launch_date_time = time_stamp_to_date_time_adjoin(launch_time_stamp)

        if not self.result_save_dir_modified:
            self.result_save_dir = os.path.join(self.result_save_dir,
                                                "{}_{}".format(self.dataset_name, self.launch_date_time))
            self.result_save_dir_modified = True
        self.log_path = os.path.join(self.result_save_dir, "logs.txt")
        self.ckp_save_dir = self.result_save_dir

        if self.optimizer is None:
            self.init_optimizer()
            self.init_scheduler(cur_epochs=self.epoch_num)

        batch_print_inter = 0
        vis_inter = math.ceil(self.epoch_num * self.configs.debug_params.vis_inter)
        ckp_save_inter = math.ceil(self.epoch_num * self.configs.debug_params.model_save_inter)

        # self.symmetry_knn_indices = self.train_loader.dataset.symmetry_knn_indices

        # if self.symmetry_knn_indices is not None and len(self.neighbor_row) == 0:
        #     for i in range(self.n_samples):
        #         self.neighbor_nums.append(len(self.symmetry_knn_indices[i]))
        #         self.neighbor_col.extend(self.symmetry_knn_indices[i])
        #         self.neighbor_row.extend(np.ones(len(self.symmetry_knn_indices[i]), dtype=np.int) * i)
        #     self.neighbor_col = np.array(self.neighbor_col, dtype=np.int)
        #     self.neighbor_row = np.array(self.neighbor_row, dtype=np.int)

        return batch_print_inter, vis_inter, ckp_save_inter

    def init_optimizer(self):
        if self.configs.method_params.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)
        elif self.configs.method_params.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9,
                                             weight_decay=0.0001)

    def init_scheduler(self, cur_epochs, base=0, gamma=0.1, milestones=None):
        if milestones is None:
            milestones = [0.8, 0.9]
        if self.configs.method_params.scheduler == "multi_step":
            self.scheduler = MultiStepLR(self.optimizer, milestones=[int(base + p * cur_epochs) for p in milestones],
                                         gamma=gamma, last_epoch=-1)
        elif self.configs.method_params.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader),
                                                                        eta_min=0.00001, last_epoch=-1)
        elif self.configs.method_params.scheduler == "cosine_warm":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=1,
                                                                                  eta_min=0.00001, last_epoch=-1)
        elif self.configs.method_params.scheduler == "on_plateau":
            # for metric
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=6, threshold=0.0005,
                                               threshold_mode='abs', cooldown=0, min_lr=0.00001, eps=1e-08,
                                               verbose=True)
            # for loss
            # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, threshold=0.0005,
            #                                    threshold_mode='rel', cooldown=5, min_lr=0.00001, eps=1e-08, verbose=True)

    def _before_epoch(self, epoch):
        self.model = self.model.to(self.device)
        training_loss = 0
        train_iterator = iter(self.train_loader)
        if self.multi:
            self.train_loader.sampler.set_epoch(epoch)
        return train_iterator, training_loss

    def _step_prepare(self, *args):
        pass

    def _train_step(self, *args):
        return None

    def _after_epoch(self, ckp_save_inter, epoch, training_loss, training_loss_history, val_inter):

        if self.configs.method_params.scheduler == "cosine" and epoch >= 10:
            self.scheduler.step()
        elif self.configs.method_params.scheduler == "multi_step":
            self.scheduler.step()

        if not self.multi or self.device_id == 0:
            train_loss = np.mean(training_loss)
            if epoch % self.print_iter == 0:
                epoch_template = 'Epoch %d/%d, Train Loss: %.5f, '
                epoch_output = epoch_template % (epoch, self.epoch_num, train_loss)
                InfoLogger.info(epoch_output)
                self.message_queue.put(epoch_output)
            # 训练损失记录
            training_loss_history.append(float(train_loss))
            embeddings, h = self.post_epoch(ckp_save_inter, epoch, val_inter)

            return embeddings
        return None

    def _train_end(self, test_loss_history, training_loss_history, embeddings):
        if self.save_final_embeddings:
            np.save(os.path.join(self.result_save_dir, "embeddings_{}.npy".format(self.epoch_num)), embeddings)

        if self.metric_tool is not None:
            self.metric_tool.join(timeout=60)

        if self.print_time_info:
            end_time = time.time()
            total_cost_time = end_time - self.sta_time
            self.sta_time = end_time
            output = "Train cost time: {} Total cost time: {}".format(self.train_time_cost, total_cost_time)
            InfoLogger.info(output)
            self.message_queue.put(output)
        self.message_queue.put("end")

        if self.save_model:
            self.save_weights(self.epoch_num)

        if self.draw_loss:
            # 画出损失函数
            x_idx = np.linspace(self.start_epoch, self.epoch_num, self.epoch_num - self.start_epoch)
            save_path = os.path.join(self.result_save_dir,
                                     "{}_loss_{}.jpg".format(self.configs.method_params.method, self.epoch_num))
            draw_loss(training_loss_history, test_loss_history, x_idx, save_path)

        self.log_process.join(timeout=5)
        shutil.copyfile(self.tmp_log_path, self.log_path)
        InfoLogger.info("Training process logging to {}".format(self.log_path))

    def train(self, launch_time_stamp=None, target_metric_val=-1):
        batch_print_inter, self.vis_inter, ckp_save_inter = self._train_begin(launch_time_stamp)
        embeddings = None
        param_dict = None
        training_loss_history = []
        test_loss_history = []
        # 迭代训练
        for epoch in range(self.start_epoch, self.epoch_num):
            # epoch准备工作
            train_iterator, training_loss = self._before_epoch(epoch)
            # epoch迭代
            for idx, data in enumerate(train_iterator):
                train_data = self._step_prepare(data, param_dict, train_iterator)
                # mini batch迭代
                loss = self._train_step(*train_data)
                training_loss += loss
            # epoch扫尾工作
            metrics, val_metrics, embeddings = self._after_epoch(ckp_save_inter, epoch + 1, training_loss,
                                                                 training_loss_history, self.vis_inter)

        self._train_end(test_loss_history, training_loss_history, embeddings)
        return embeddings

    def quantitative_test_preprocess(self, embedding_data=None, device='cuda'):

        data = torch.tensor(self.train_loader.dataset.get_all_data()).to(device).float()

        if self.is_image:
            data = data / 255.
        if embedding_data is None:
            self.model.to(device)
            embedding_data, _ = self.acquire_latent_code_allin(data, device)

        if self.metric_tool is None:
            self.build_metric_tool()

        return embedding_data, self.fixed_k, data

    def build_metric_tool(self):
        eval_used_data = self.train_loader.dataset.get_all_data()
        eval_used_data = np.reshape(eval_used_data, (eval_used_data.shape[0], np.product(eval_used_data.shape[1:])))

        cache_path = ConfigInfo.NEIGHBORS_CACHE_DIR.format(self.dataset_name, self.fixed_k)
        pair_cache_path = ConfigInfo.PAIRWISE_DISTANCE_DIR.format(self.dataset_name)
        knn_indices, knn_dists = compute_knn_graph(eval_used_data, cache_path, self.fixed_k,
                                                   pair_cache_path, accelerate=True)

        preload = eval_used_data.shape[0] <= 30000
        pairwise_distance = get_pairwise_distance(eval_used_data, "euclidean", pair_cache_path, preload)

        self.metric_tool = MetricProcess(self.queue_set, self.message_queue, self.dataset_name, eval_used_data,
                                         self.train_loader.dataset.targets,
                                         knn_indices, knn_dists, pairwise_distance,
                                         self.result_save_dir, norm=self.is_image, k=self.fixed_k)

        self.metric_tool.start()

    def save_weights(self, epoch, prefix_name=None):
        if prefix_name is None:
            prefix_name = epoch
        if not os.path.exists(self.ckp_save_dir):
            os.mkdir(self.ckp_save_dir)
        weight_save_path = os.path.join(self.ckp_save_dir, "{}_{}.pth.tar".
                                        format(self.configs.method_params.method, prefix_name))
        torch.save({'epoch': epoch, 'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr': self.lr, 'launch_time': self.launch_date_time}, weight_save_path)
        InfoLogger.info("weights successfully save to {}".format(weight_save_path))

    def quantitative_test_all(self, epoch, embedding_data=None, mid_embeddings=None, device='cuda', val=False):
        embedding_data, k, _ = self.quantitative_test_preprocess(embedding_data, device)
        # 向评估进程传输评估数据

        if val:
            self.queue_set.test_eval_data_queue.put(
                [epoch, k, embedding_data, (epoch == self.epoch_num), False])
        else:
            self.queue_set.eval_data_queue.put([epoch, k, embedding_data, (epoch == self.epoch_num), False])

    def quantitative_eval_data(self):
        self.preprocess()
        self.build_metric_tool()
        class_num, cls_count_str, is_balanced, sparsity_ratio, intrinsic_dim, origin_neighbor_hit, \
        n_samples, n_dims = self.metric_tool.eval_dataset(
            self.n_neighbors)
        metric_template = "Is Balanced: %s Sparsity Ratio: %.4f Intrinsic Dimension: %d " \
                          "Origin Neighbor Hit(label): %.4f"
        output = metric_template % (is_balanced, sparsity_ratio, intrinsic_dim, origin_neighbor_hit)
        self.message_queue.put(output)
        InfoLogger.info(output)
        return is_balanced, sparsity_ratio, intrinsic_dim, origin_neighbor_hit

    def load_checkpoint(self, checkpoint_path):
        model_CKPT = torch.load(checkpoint_path)
        self.model.load_state_dict(model_CKPT['state_dict'])
        InfoLogger.info('loading checkpoint success!')
        if self.optimizer is None:
            self.init_optimizer()
        self.optimizer.load_state_dict(model_CKPT['optimizer'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def load_weights(self, checkpoint_path, train):
        self.preprocess(train)
        model_CKPT = torch.load(checkpoint_path)
        self.model.load_state_dict(model_CKPT['state_dict'])
        self.init_optimizer()
        self.optimizer.load_state_dict(model_CKPT['optimizer'])
        self.optimizer.param_groups[0]['lr'] = self.lr
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        return model_CKPT

    def load_weights_train(self, resume_epochs, checkpoint_path):
        self.resume_epochs = resume_epochs
        model_CKPT = self.load_weights(checkpoint_path)
        self.start_epoch = model_CKPT["epoch"]
        self.launch_date_time = model_CKPT["launch_time"]
        self.train()

    def load_weights_visualization(self, checkpoint_path, vis_save_path, device='cuda', train=False):
        self.preprocess(train, load_data=True)
        self.load_checkpoint(checkpoint_path)
        embeddings, val_embeddings = self.visualize(vis_save_path, device=device)
        return embeddings

    def load_weights_cal_embeddings(self, checkpoint_path, device='cuda'):
        self.preprocess()
        self.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device)
        data = torch.tensor(self.train_loader.dataset.get_all_data()).to(device).float()
        embeddings = self.cal_lower_embeddings(data)
        return embeddings

    def train_for_visualize(self):
        if not self.multi or self.device_id == 0:
            InfoLogger.info("Start train for Visualize")
        launch_time_stamp = int(time.time())
        self.preprocess()
        self.pre_embeddings = self.train(launch_time_stamp)
        return self.pre_embeddings

    def cal_lower_embeddings(self, data):
        if self.is_image:
            data = data / 255.
        embeddings = self.acquire_latent_code_allin(data, self.device)
        return embeddings

    def visualize(self, vis_save_path=None, device="cuda"):
        # InfoLogger.info("Start Visualization")
        self.model.to(device)

        data = torch.tensor(self.train_loader.dataset.get_all_data()).to(device).float()
        # val_data = torch.tensor(self.test_loader.dataset.get_all_data()).to(device).float()
        z = self.cal_lower_embeddings(data)
        val_z = None
        # val_z = self.cal_lower_embeddings(val_data)

        # z = self.acquire_latent_code_in_batch(data)
        # val_z = self.acquire_latent_code_in_batch(val_data)

        if self.configs.exp_params.latent_dim <= 2:
            draw_projections(z, self.train_loader.dataset.targets, vis_save_path)

        return z, val_z

    def acquire_latent_code_allin(self, data, device):
        with torch.no_grad():
            """
            BN层中的running_mean和running_var的更新是在forward()操作中进行的，而不是optimizer.step()中进行的，
            因此如果处于训练状态，就算你不进行手动step()，BN的统计特性也会变化的。当以batch的方式进行推理时，就会一直更新这些参数
            使用 model.eval() 将模型切换到测试模式，此时BN和Dropout中的参数不会改变
            """
            self.model.eval()
            # self.model.encoder.eval()
            if self.multi:
                z = self.model.module.acquire_latent_code(data)
            else:
                z = self.model.acquire_latent_code(data)
            self.model.train()

            z = z.cpu().numpy()
        return z

    def preprocess(self, train=True, load_data=True):
        if load_data:
            self.build_dataset()
        if train and (not self.multi or self.device_id == 0):
            self.log_process = LogWriter(self.tmp_log_path, self.log_path, self.message_queue)
            self.log_process.start()
        self.model_prepare()
        if self.multi:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.configs.local_rank],
                                                                   output_device=self.configs.local_rank)

    def model_prepare(self):
        pass

    def build_dataset(self):
        pass

    def load_weights_quantitative_test_all(self, checkpoint_path=None, device='cuda'):
        self.preprocess()
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        self.quantitative_test_all(0, device=device)

    def post_epoch(self, ckp_save_inter, epoch, vis_inter):
        # 随机采样验证
        val_metrics = None
        embeddings = None
        h = None
        val_h = None
        val_embeddings = None
        # 可视化结果验证
        vis_save_path = os.path.join(self.result_save_dir, '{}_vis_{}.jpg'.format(self.dataset_name, epoch))
        final = epoch == self.epoch_num

        if epoch % vis_inter == 0 and self.do_vis:
            # 随机采样、对比验证
            if not os.path.exists(self.result_save_dir):
                os.makedirs(self.result_save_dir)
                if self.config_path is not None:
                    shutil.copyfile(self.config_path, os.path.join(self.result_save_dir, "config.yaml"))

            embeddings, val_embeddings = self.visualize(vis_save_path, device=self.device)

        # 定量指标计算
        if ((self.test_inter > 0 and epoch % self.test_inter == 0) or
            (self.metric_tool is not None and epoch == 1 and self.metric_tool.record_lost_neighbors)) or final:
            # 创建结果文件夹
            if not os.path.exists(self.result_save_dir):
                os.makedirs(self.result_save_dir)
                if self.config_path is not None:
                    shutil.copyfile(self.config_path, os.path.join(self.result_save_dir, "config.yaml"))

            if embeddings is None:
                if self.do_vis:
                    embeddings, val_embeddings = self.visualize(vis_save_path, device=self.device)
                else:
                    self.model.to(self.device)
                    data = torch.tensor(self.train_loader.dataset.get_all_data()).to(self.device).float()
                    embeddings = self.cal_lower_embeddings(data)

            if self.do_test:
                # 训练集上进行定量测试
                self.quantitative_test_all(epoch, embeddings, h, device=self.device)
            # 如果没有使用下采样训练，在测试集上进行定量测试
            # self.quantitative_test_all(epoch, val_embeddings, val_h, device=self.device, val=True)

        # 保存模型权重
        if epoch % ckp_save_inter == 0 and self.test_inter == 0:
            if not os.path.exists(self.ckp_save_dir):
                os.mkdir(self.ckp_save_dir)
            self.save_weights(epoch)

        return embeddings, h

    def average_metrics_test(self, test_time):
        save_path = os.path.join(self.result_save_dir, "{}_{}_{}_results.xlsx".format(self.dataset_name,
                                                                                      "umap",
                                                                                      self.epoch_num))
        val_save_path = save_path.replace("_results", "_val_results")
        metric_num = len(METRIC_NAMES)
        results = np.empty((test_time, metric_num))
        val_results = np.empty((test_time, metric_num))
        metric_valid = False

        time_costs = []

        for i in range(test_time):
            InfoLogger.info("Start {}/{} train!".format(i + 1, test_time))
            launch_time_stamp = int(time.time())
            self.eval_time_cost = 0
            self.model = MODELS[self.configs.method_params.method](self.configs, device=self.device)
            self.model = self.model.to(self.device)
            self.start_epoch = 0
            self.queue_set = QueueSet()
            self.message_queue = Queue()
            self.preprocess()
            if i > 0:
                self.build_metric_tool()
                # self.build_val_metric_tool()
            self.optimizer = None
            self.launch_date_time = None
            embeddings = self.train(launch_time_stamp)
            if not self.multi or self.device_id == 0:
                metrics = self.metric_tool.queue_set.eval_result_queue.get()
                time_costs.append(self.train_time_cost)
                # val_metrics = self.metric_tool.queue_set.test_eval_result_queue.get()
                if metrics is not None and len(metrics) > 0:
                    results[i] = metrics
                    # val_results[i] = val_metrics
                    metric_valid = True
            # if self.configs.exp_params.refine:
            #     self.refiner = None

        if metric_valid and (not self.multi or self.device_id == 0):
            return results_to_excel(METRIC_NAMES, metric_num, results, time_costs, save_path)
            # self.results_to_excel(metric_names, metric_num, val_results, val_save_path)

    def load_embeddings_eval(self, embeddings, val=False):
        self.preprocess()
        self.quantitative_test_preprocess(embeddings)
        metric_tool = self.val_metric_tool if val else self.metric_tool

        trust, continuity, neighbor_hit, knn_ac, ka_5, ka_1, sim_fake_ratio, dissim_lost_ratio, sc, dsc, gon \
            = metric_tool.cal_all_metrics(self.fixed_k, embeddings, final=True)

        metric_template = "Trust: %.4f Continuity: %.4f Neighbor Hit: %.4f KA(10): %.4f KA(5): %.4f KA(11): %.4f " \
                          "SC: %.4f DSC: %.4f GON: %.4f Sim Fake: %.4f Dissim Lost: %.4f "
        metric_output = metric_template % (trust, continuity, neighbor_hit, knn_ac, ka_5, ka_1, sc, dsc, gon,
                                           sim_fake_ratio, dissim_lost_ratio)

        if val:
            metric_output = "Validation " + metric_output

        InfoLogger.info(metric_output)
        self.message_queue.put(metric_output)

        return trust, continuity, neighbor_hit, knn_ac, ka_5, ka_1, sim_fake_ratio, dissim_lost_ratio, sc, dsc, gon
