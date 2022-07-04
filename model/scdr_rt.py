import time
from multiprocessing import Queue, Process

import numpy as np

from dataset.warppers import eval_knn_acc
from experiments.scdr_trainer import SCDRTrainer
from model.scdr import SCDRBase
from utils.metrics_tool import metric_neighbor_preserve_introduce
from utils.nn_utils import StreamingKNNSearchApprox, compute_knn_graph, StreamingKNNSearchApprox2
from utils.scdr_utils import KeyPointsGenerater, DataSampler


class RTSCDRModel(SCDRBase):
    def __init__(self, shift_buffer_size, n_neighbors, model_trainer: SCDRTrainer, initial_train_num,
                 initial_train_epoch, finetune_epoch, finetune_data_ratio=1.0, ckpt_path=None):
        SCDRBase.__init__(self, n_neighbors, model_trainer, initial_train_num, initial_train_epoch, finetune_epoch,
                          finetune_data_ratio, ckpt_path)

        self.shift_buffer_size = shift_buffer_size
        self.initial_data_buffer = None
        self.initial_label_buffer = None

        # 这种情况会导致算法检测不准确，所以检测出异常点的可能性更高（因为高维分布的数据少了），
        # 会导致投影函数更新的次数增加。
        self.key_data_rate = 0.5
        # self.key_data_rate = 1.0
        self.using_key_data_in_lof = True

        self.data_sampler = DataSampler(self.n_neighbors, finetune_data_ratio, self.minimum_finetune_data_num,
                                        time_based_sample=False, metric_based_sample=False)

        # 没有经过模型拟合的数据
        self.unfitted_data = None

        # self.knn_searcher_approx = StreamingKNNSearchApprox()
        self.knn_searcher_approx = StreamingKNNSearchApprox2()

    def fit_new_data(self, data, labels=None):
        new_data_num = data.shape[0]
        if self.using_key_data_in_lof:
            self._generate_key_points(data)

        self.data_sampler.update_sample_weight(data.shape[0])

        if not self.model_trained:
            if not self._caching_initial_data(data, labels):
                return None
            self._initial_project(self.initial_data_buffer, self.initial_label_buffer)
            # TODO: 调试用，调试完成后删除
            sta = time.time()
            self.knn_searcher.search(self.initial_data_buffer, self.n_neighbors, query=False)
            self.knn_cal_time += time.time() - sta
        else:
            fit_data = self.key_data if self.using_key_data_in_lof else self.dataset.total_data
            self._detect_distribution_shift(fit_data, data, labels)
            self.data_num_list.append(new_data_num + self.data_num_list[-1])

            # 使用之前的投影函数对最新的数据进行投影
            sta = time.time()
            data_embeddings = self.model_trainer.infer_embeddings(data).numpy().astype(float)
            data_embeddings = np.reshape(data_embeddings, (new_data_num, self.pre_embeddings.shape[1]))
            self.model_repro_time += time.time() - sta
            cur_total_embeddings = np.concatenate([self.pre_embeddings, data_embeddings], axis=0, dtype=float)

            # 更新knn graph
            """
                刚开始时的精度还可以，但是当投影函数在没有更新的情况下投影的新数据越来越多，精度就会大幅下降。模型更新之后，精度又会上升。
                这主要是因为当旧投影函数所投影的数据越来越多之后，投影中的错误就越多了（旧投影函数直接投影的这些点的邻域保持是比较差的），
                所以精度下降。
                那是不是可以在模型拟合过的数据上使用近似，然后在没有拟合过的数据上使用精确的方法.
                1. 首先用近似的方法求出一部分近邻点下标
                2. 然后与没有参与训练模型的新数据合并
                3. 在合并后的数据中使用annoy进行准确的搜索
                
                此外， 当数据量越来越多的时候，低维空间中候选的数据也应该越来越多，这样才能保证精度。可以考虑使用聚类的方法
                选择最接近的两个聚类中的数据作为候选数据。
            """
            sta = time.time()
            # approx_nn_indices, approx_nn_dists = self.knn_searcher_approx.search(self.n_neighbors, self.pre_embeddings,
            #                                                                      self.dataset.total_data,
            #                                                                      data_embeddings, data,
            #                                                                      self.pre_model_update)

            nn_indices, nn_dists = self.knn_searcher_approx. \
                search(self.n_neighbors, self.pre_embeddings[:self.fitted_data_num],
                       self.dataset.total_data[:self.fitted_data_num], data_embeddings, data, self.unfitted_data)

            self.knn_approx_time += time.time() - sta

            # TODO: 调试用，调试完成后删除
            # sta = time.time()
            # nn_indices, nn_dists = self.knn_searcher.search(data, self.n_neighbors)
            # self.knn_cal_time += time.time() - sta
            # acc_list = eval_knn_acc(nn_indices, approx_nn_indices)

            self.dataset.add_new_data(data, nn_indices, nn_dists, labels)
            # self.dataset.add_new_data(data, approx_nn_indices, approx_nn_dists, labels)

            if len(self.cached_shift_indices) <= self.shift_buffer_size:
                # TODO：如何计算这些最新数据的投影结果，以保证其在当前时间点是可信的，应该被当作异常点/离群点进行处理
                # TODO: 这边即使数据的分布没有发生变化，但是在投影质量下降到一定程度后，也要进行模型更新
                self.pre_embeddings = cur_total_embeddings
                self.unfitted_data = data if self.unfitted_data is None else np.concatenate([self.unfitted_data, data],
                                                                                            axis=0)
            else:
                self.pre_embeddings = self._adapt_distribution_change(cur_total_embeddings)
        # self._gather_data_stream()
        return self.pre_embeddings

    def _adapt_distribution_change(self, *args):
        # TODO:先还是展示当前投影的结果，然后在后台更新投影模型，模型更新完成后，再更新所有数据的投影
        self.pre_embeddings = self._update_projection_model()
        self.cached_shift_indices = None
        self.unfitted_data = None

    def _sample_training_data(self, *args):
        pre_n_samples = self.pre_embeddings.shape[0]
        must_indices = np.concatenate([self.cached_shift_indices, np.arange(self.fitted_data_num,
                                                                            self.dataset.cur_n_samples, 1)])
        return self.data_sampler.sample_training_data(self.dataset.total_data[:pre_n_samples], self.pre_embeddings,
                                                      self.dataset.knn_indices[:pre_n_samples], must_indices)

    def _caching_initial_data(self, data, labels):
        self.initial_data_buffer = data if self.initial_data_buffer is None \
            else np.concatenate([self.initial_data_buffer, data], axis=0)
        if labels is not None:
            self.initial_label_buffer = labels if self.initial_label_buffer is None \
                else np.concatenate([self.initial_label_buffer, labels], axis=0)

        return self.initial_data_buffer.shape[0] >= self.initial_train_num

    def buffer_empty(self):
        return self.cached_shift_indices is None

    def clear_buffer(self, final_embed=False):
        # 更新knn graph
        self.pre_embeddings = self._update_projection_model()
        return self.pre_embeddings

    def _generate_key_points(self, data):
        if self.key_data is None:
            self.key_data = KeyPointsGenerater.generate(data, self.key_data_rate)[0]
        else:
            key_data = KeyPointsGenerater.generate(data, self.key_data_rate)[0]
            self.key_data = np.concatenate([self.key_data, key_data], axis=0)


class RTSCDRParallel(RTSCDRModel):
    def __init__(self, queue_set, shift_buffer_size, n_neighbors, model_trainer: SCDRTrainer, initial_train_num,
                 initial_train_epoch, finetune_epoch, finetune_data_ratio=1.0, ckpt_path=None):
        RTSCDRModel.__init__(self, shift_buffer_size, n_neighbors, model_trainer, initial_train_num,
                             initial_train_epoch, finetune_epoch, finetune_data_ratio, ckpt_path)
        self.queue_set = queue_set
        self.scdr_trainer_process = None

    def _initial_project(self, data, labels=None):
        super()._initial_project(data, labels)
        self.scdr_trainer_process = SCDRTrainerProcess(self, self.queue_set)
        self.scdr_trainer_process.start()

    def re_embedding_all(self, data):
        data_embeddings = self.model_trainer.infer_embeddings(data).numpy().astype(float)
        self.pre_embeddings = data_embeddings
        return data_embeddings

    def _adapt_distribution_change(self, *args):
        cur_total_embeddings = args[0]
        cur_data_num = cur_total_embeddings.shape[0]
        self.unfitted_data = None
        # TODO: debug检查一下，确定没有浅引用问题
        cached_shift_indices = self.cached_shift_indices
        self.cached_shift_indices = None
        data_num_list = self.data_num_list
        self.data_num_list = [0]
        fitted_data_num = self.fitted_data_num
        self.fitted_data_num = self.dataset.cur_n_samples

        # TODO:先还是展示当前投影的结果，然后在后台更新投影模型，模型更新完成后，再更新所有数据的投影

        sampled_indices = self._sample_training_data(self.time_based_sample, self.metric_based_sample,
                                                     cached_shift_indices)

        self.queue_set.data_queue.put([sampled_indices, fitted_data_num, data_num_list, cur_data_num])

        return cur_total_embeddings

    def parallel_code(self, sampled_indices, fitted_data_num, data_num_list, cur_data_num):
        # 下面这些应该并行
        # 更新knn graph
        sta = time.time()
        self.dataset.update_knn_graph(self.dataset.total_data[:fitted_data_num],
                                      self.dataset.total_data[fitted_data_num:], data_num_list, cur_data_num)
        self.knn_update_time += time.time() - sta

        self.model_trainer.update_batch_size(len(sampled_indices))

        self.model_trainer.update_dataloader(self.finetune_epoch, sampled_indices)

        sta = time.time()
        # TODO:设置一个队列，当模型更新进程把模型更新好之后就将其加入到该队列中，然后投影进程监听到该队列中有数据就替换当前模型
        self.model_trainer.resume_train(self.finetune_epoch)
        self.model_update_time += time.time() - sta


class SCDRTrainerProcess(Process):
    def __init__(self, rt_scdr, queue_set):
        self.name = "SCDR模型更新进程"
        Process.__init__(self, name=self.name)
        self.queue_set = queue_set
        self.rt_scdr = rt_scdr

    def run(self) -> None:
        while True:
            # 获取训练相关的数据，现在的问题在于tensor数据不能跨进程传输
            training_info = self.queue_set.data_queue.get()
            print("准备更新模型！")
            self.rt_scdr.parallel_code(*training_info)
            self.queue_set.re_embedding_flag_queue.put(True)

