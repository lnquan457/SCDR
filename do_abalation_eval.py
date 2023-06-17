import os
import time

import h5py
import numpy as np
import torch

from dataset.warppers import StreamingDatasetWrapper
from do_eval import cal_knn
from model.dr_models.ModelSets import MODELS
from model.scdr.data_processor import query_knn2
from model.scdr.dependencies.experiment import draw_projections
from model.scdr.dependencies.scdr_utils import ClusterRepDataSampler
from model.scdr.model_trainer import SCDRTrainer
from scripts.making_streaming_data import check_path_exists
from stream_main import parse_args
from utils.common_utils import get_config, time_stamp_to_date_time_adjoin
from utils.constant_pool import ConfigInfo
from utils.metrics_tool import Metric
from utils.nn_utils import get_pairwise_distance, compute_knn_graph


def build_metric_tool(dataset_name, data, label, eval_k):
    pairwise_distance = get_pairwise_distance(data)

    knn_indices, knn_dists = cal_knn(pairwise_distance, eval_k)
    metric_tool = Metric(dataset_name, data, label, knn_indices, knn_dists, pairwise_distance, k=eval_k)
    return metric_tool


def sample_rep_data(cluster_rep_data_sampler, s_dataset, total_embeddings, fitted_num, n_neighbors, pre_knn_indices):
    embeddings = total_embeddings[:fitted_num]
    sta = time.time()
    ravel_1 = np.reshape(np.repeat(total_embeddings[:, np.newaxis, :], n_neighbors // 2, 1), (-1, 2))
    ravel_2 = total_embeddings[np.ravel(pre_knn_indices[:, :n_neighbors // 2])]

    embedding_nn_dist = np.mean(np.linalg.norm(ravel_1 - ravel_2, axis=-1))
    rep_batch_nums, rep_data_indices, cluster_indices, _, _ = \
        cluster_rep_data_sampler.sample(embeddings, eps=embedding_nn_dist, min_samples=n_neighbors,
                                        labels=s_dataset.get_total_label()[:fitted_num])
    return cluster_indices, rep_batch_nums, rep_data_indices


if __name__ == '__main__':
    '''
        0. 以PD情况为基准
        1. 首先使用初始数据集训练模型
        2. 每次按顺序选batch_num（1000）个数据
        3. 与当前数据合并，计算新数据的kNN关系，更新旧数据的kNN关系
        4. 采样部分旧数据作为代表数据
        5. 使用代表数据和新数据一起对模型进行增量式更新
            5.1 使用质量约束 + 稳定性约束
            5.2 只使用质量约束
            5.3 只使用稳定性约束
        6. 使用当前模型得到初始数据的嵌入结果，以及当前新数据的嵌入结果
        7. 分别对初始数据的嵌入结果以及新数据的嵌入结果进行评估；
        8. 重复步骤2-7执行3次或者5次，统计旧数据上的评估结果，新数据上的评估结果
    '''
    situation = "PD"
    dataset_list = ["arem", "basketball", "shuttle", "HAR_2", "covid_twi", "mnist_fla"]
    dim_list = [6, 6, 9, 561, 256, 784]
    metric_list = ["Trust", "Continuity", "Neighbor Hit", "KA(10)", "Shepard Goodness", "DEMaP", "Position Change"]
    batch_num = 1000
    inc_time = 5
    finetune_epochs = 50
    eval_k = 10

    data_dir = r"D:\Projects\流数据\Data\H5 Data"
    indices_dir = r"D:\Projects\流数据\Data\new\indices_seq"
    save_dir = r"D:\Projects\流数据\Evaluation\消融实验\质量约束"
    device = "cuda:0"
    args = parse_args()
    cfg_path = ConfigInfo.MODEL_CONFIG_PATH.format(args.method)
    cfg = get_config()
    cfg.merge_from_file(cfg_path)

    for i, dataset in enumerate(dataset_list):
        # if i < 2:
        #     continue
        cfg.exp_params.input_dims = dim_list[i]
        with h5py.File(os.path.join(data_dir, "{}.h5".format(dataset)), "r") as hf:
            x = np.array(hf['x'], dtype=float)
            y = np.array(hf['y'], dtype=int)

        initial_indices, after_indices = np.load(os.path.join(indices_dir, "{}_{}.npy".format(dataset, situation)),
                                                 allow_pickle=True)
        initial_data = x[initial_indices]
        initial_label = y[initial_indices]

        stream_data = x[after_indices]
        stream_label = y[after_indices]

        stream_dataset = StreamingDatasetWrapper(initial_data.shape[0] / 10, 10, 100000, device)
        stream_dataset.add_new_data(data=initial_data, labels=initial_label)

        res_save_dir = os.path.join(save_dir, dataset,
                                    time_stamp_to_date_time_adjoin(int(time.time())))

        cdr_model = MODELS["LwF_CDR"](cfg, device=device)
        trainer = SCDRTrainer(cdr_model, dataset, cfg, res_save_dir, cfg_path, device)
        trainer.result_save_dir = res_save_dir
        initial_model_save_path = os.path.join(save_dir, dataset, "initial_data_ckpt.pth.tar")

        if os.path.exists(initial_model_save_path):
            trainer.model = trainer.model.to(device)
            trainer.load_weights(initial_model_save_path, train=True)
            initial_embeddings_0 = trainer.cal_lower_embeddings(initial_data)
            trainer.initialize_streaming_dataset(stream_dataset)
            trainer.update_batch_size(stream_dataset.get_n_samples())
            trainer.update_dataloader(trainer.initial_train_epoch)
        else:
            initial_embeddings_0 = trainer.first_train(stream_dataset)
            # 保存初始数据集上的模型便于之后复用
            torch.save({'epoch': trainer.epoch_num, 'state_dict': trainer.model.state_dict(),
                        'optimizer': trainer.optimizer.state_dict(),
                        'lr': trainer.lr, 'launch_time': trainer.launch_date_time}, initial_model_save_path)

        # 绘制并保存散点图
        vis_save_dir = os.path.join(res_save_dir, "vis")
        check_path_exists(vis_save_dir)
        draw_projections(initial_embeddings_0, initial_label, os.path.join(vis_save_dir, "vis_init_0.jpg"))
        metric_log_file = open(os.path.join(res_save_dir, "metric_log.txt"), "w")

        # 保存嵌入结果
        npy_save_dir = os.path.join(res_save_dir, "embeddings")
        check_path_exists(npy_save_dir)
        np.save(os.path.join(npy_save_dir, "embedding_0.npy"), initial_embeddings_0, allow_pickle=True)

        # # 需要进行评估
        # metric_tool = build_metric_tool(dataset, initial_data, initial_label)
        # metric_res = metric_tool.cal_simplified_metrics(eval_k, initial_embeddings_0, knn_k=eval_k)
        # metric_log_file.write("Step 0")
        # for j, metric_name in enumerate(metric_list):
        #     metric_log_file.write("%s: %.4f" % (metric_name, metric_res[j]))
        #
        # metric_log_file.write("\n")

        pre_n_samples = initial_data.shape[0]
        pre_embeddings = initial_embeddings_0

        cluster_rep_data_sampler = ClusterRepDataSampler(0.07, 50, True)

        stream_idx = 0
        for j in range(inc_time):
            cur_stream_data = stream_data[stream_idx:stream_idx + batch_num]
            cur_stream_label = stream_label[stream_idx:stream_idx + batch_num]
            stream_idx += batch_num

            pre_knn_indices = stream_dataset.get_knn_indices()
            total_data = np.concatenate([trainer.stream_dataset.get_total_data(), cur_stream_data], axis=0)
            newest_knn_indices, newest_knn_dists = query_knn2(total_data, total_data, k=trainer.n_neighbors)

            # 需要更新kNN，更新trainer的训练数据
            trainer.stream_dataset.add_new_data(cur_stream_data, None, cur_stream_label,
                                                newest_knn_indices[-batch_num:], newest_knn_dists[-batch_num:])

            # trainer.stream_dataset._knn_manager.update_knn_graph(newest_knn_indices, newest_knn_dists)
            # trainer.stream_dataset.raw_knn_weights = np.concatenate([trainer.stream_dataset.raw_knn_weights,
            #                                                          np.ones((batch_num, trainer.n_neighbors))],
            #                                                         axis=0)
            trainer.stream_dataset._sigmas = np.concatenate([trainer.stream_dataset._sigmas, np.ones(batch_num)])
            trainer.stream_dataset._rhos = np.concatenate([trainer.stream_dataset._rhos, np.ones(batch_num)])

            trainer.stream_dataset.raw_knn_weights = np.concatenate([trainer.stream_dataset.raw_knn_weights,
                                                                     np.ones(shape=(batch_num, trainer.n_neighbors))])
            trainer.stream_dataset.symmetric_nn_indices = np.concatenate([trainer.stream_dataset.symmetric_nn_indices,
                                                                          np.ones(batch_num)])
            trainer.stream_dataset.symmetric_nn_weights = np.concatenate([trainer.stream_dataset.symmetric_nn_weights,
                                                                          np.ones(batch_num)])

            knn_indices, knn_dists = compute_knn_graph(trainer.stream_dataset.get_total_data(), None,
                                                       trainer.n_neighbors, None)
            trainer.stream_dataset._knn_manager.update_knn_graph(knn_indices, knn_dists)

            trainer.stream_dataset._cached_neighbor_change_indices = np.arange(pre_n_samples + batch_num)
            trainer.stream_dataset.update_cached_neighbor_similarities()

            sample_indices = np.arange(pre_n_samples, pre_n_samples + batch_num)

            trainer.prepare_resume(pre_n_samples, len(sample_indices), finetune_epochs, sample_indices)
            steady_constraints = trainer.stream_dataset.cal_old2new_relationship(old_n_samples=pre_n_samples)

            cluster_indices, rep_batch_nums, rep_data_indices = sample_rep_data(cluster_rep_data_sampler,
                                                                                trainer.stream_dataset, pre_embeddings,
                                                                                pre_n_samples, trainer.n_neighbors,
                                                                                pre_knn_indices)
            pre_rep_data_info = [rep_batch_nums, rep_data_indices, cluster_indices, pre_n_samples, steady_constraints]

            # 需要更新训练数据
            cur_new_embeddings = trainer.resume_train(finetune_epochs, pre_rep_data_info)

            # 绘制并保存旧数据和新数据的散点图
            cur_initial_embeddings = trainer.cal_lower_embeddings(initial_data)
            cur_total_embeddings = trainer.cal_lower_embeddings(trainer.stream_dataset.get_total_data())
            trainer.stream_dataset.add_new_data(embeddings=cur_new_embeddings)

            # 绘制并保存散点图
            draw_projections(cur_total_embeddings, trainer.stream_dataset.get_total_label(),
                             os.path.join(vis_save_dir, "vis_init_{}.jpg".format(j+1)))

            # 保存嵌入结果
            npy_save_dir = os.path.join(res_save_dir, "embeddings")
            np.save(os.path.join(npy_save_dir, "embedding_{}.npy".format(j+1)), cur_total_embeddings, allow_pickle=True)

            pre_embeddings = cur_total_embeddings
            pre_n_samples = pre_embeddings.shape[0]