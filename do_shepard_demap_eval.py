import os
import time
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from dataset.warppers import KNNManager
from experiments.streaming_experiment import check_path_exist
from utils.constant_pool import METRIC_NAMES
from utils.metrics_tool import Metric, cal_global_position_change, knn_score
from utils.nn_utils import compute_knn_graph, get_pairwise_distance
import matplotlib.pyplot as plt


def simple_draw(x_indices, y_indices, title, save_path):
    plt.figure()
    plt.title(title)
    plt.plot(x_indices, y_indices)
    plt.ylabel(title)
    # plt.legend()
    plt.savefig(save_path)
    plt.show()


def cal_knn(pdist):
    sorted_indices = np.argsort(pdist, axis=1)
    knn_indices = sorted_indices[:, 1:eval_k + 1]
    knn_distances = []
    for p in range(knn_indices.shape[0]):
        knn_distances.append(pdist[p, knn_indices[p]])
    knn_dists = np.array(knn_distances)
    return knn_indices, knn_dists


if __name__ == '__main__':
    situation = "ND"
    res_dir = r"D:\Projects\流数据\Evaluation\原始数据\{}\0224".format(situation)
    data_dir = r"D:\Projects\流数据\Data\H5 Data"
    indices_dir = r"D:\Projects\流数据\Data\new\indices_seq"
    save_dir = r"D:\Projects\流数据\Evaluation\原始数据\{}\0224".format(situation)
    eval_k = 10
    window_size = 5000
    # valid_metric_indices = [0, 1, 2, 3, 4]
    valid_metric_indices = [0, 1]
    total_res_data = np.zeros((5, 10, 5))
    xtreaming_buffer_size = 200
    eval_step = 100

    method_list = ["sPCA", "Xtreaming", "SIsomap++"]
    # method_list = ["sPCA"]
    # method_list = ["SCDR"]
    metric_list = ["Shepard Goodness", "DEMaP", ]
    dataset_list = ["arem", "basketball", "HAR_2", "shuttle", "mnist_fla"]

    for i, method in enumerate(method_list):
        print("Method:", method)
        method_dir = os.path.join(res_dir, method)
        j = 0
        for dataset in dataset_list:
            print("==========Dataset:", dataset)
            # if method == "sPCA" and dataset == "mnist_fla":
            #     for k, item in enumerate(valid_metric_indices):
            #         total_res_data[k, j, i] = 0
            #     j += 1
            #     continue

            dataset_dir = os.path.join(method_dir, dataset)
            with h5py.File(os.path.join(data_dir, "{}.h5".format(dataset)), "r") as hf:
                x = np.array(hf['x'], dtype=float)
                y = np.array(hf['y'], dtype=int)

            initial_indices, after_indices = np.load(os.path.join(indices_dir, "{}_{}.npy".format(dataset, situation)), allow_pickle=True)
            initial_data = x[initial_indices]
            initial_label = y[initial_indices]

            stream_data = x[after_indices]
            stream_label = y[after_indices]
            total_raw_data = np.concatenate([initial_data, stream_data], axis=0)
            total_raw_label = np.concatenate([initial_label, stream_label])

            dataset_res = []
            res_dict = {}
            # dataset_dir = r"D:\Projects\流数据\Code\SCDR\results\sPCA\arem"
            for k, item in enumerate(os.listdir(dataset_dir)):
                if item.endswith("xlsx"):
                    continue
                item_dir = os.path.join(dataset_dir, item)
                eval_embedding_dir = os.path.join(item_dir, "eval_embeddings")
                # eval_embedding_dir = r"D:\Projects\流数据\Code\SCDR\results\sPCA\arem\20230228_14h22m40s\eval_embeddings"

                metric_records = [[] for i in range(len(metric_list))]
                metric_image_dir = os.path.join(item_dir, "metric_imgs")
                check_path_exist(metric_image_dir)
                sta_time_step = 1
                eval_num = 0

                pre_pairwise_dist = None
                pre_time_step = sta_time_step
                eval_time = 0
                data_idx = len(initial_indices)
                total_data = None
                total_label = None
                while True:
                    sta_time_step += eval_step
                    jtem = "t{}.npy".format(sta_time_step - 1)
                    if not os.path.exists(os.path.join(eval_embedding_dir, jtem)):
                        break
                    cur_time_step, cur_embeddings, pre_valid_embeddings, cur_valid_embeddings = \
                        np.load(os.path.join(eval_embedding_dir, jtem), allow_pickle=True)

                    new_data_num = cur_time_step - pre_time_step
                    new_data = total_raw_data[data_idx:data_idx+new_data_num]
                    new_label = total_raw_label[data_idx:data_idx+new_data_num]
                    if total_data is None:
                        valid_init_num = window_size - new_data_num + 1
                        total_data = np.concatenate([initial_data[-valid_init_num:], new_data], axis=0)
                        total_label = np.concatenate([initial_label[-valid_init_num:], new_label])
                    else:
                        total_data = np.concatenate([total_data, new_data], axis=0)
                        total_label = np.concatenate([total_label, new_label])

                    out_num = max(0, total_data.shape[0] - window_size) - 1
                    if out_num > 0:
                        if pre_pairwise_dist is not None:
                            pre_pairwise_dist = pre_pairwise_dist[out_num:, :][:, out_num:]
                        total_data = total_data[out_num:]
                        total_label = total_label[out_num:]

                    cur_data_num = total_data.shape[0]
                    cur_embeddings = cur_embeddings[-cur_data_num:]
                    pre_valid_embeddings = pre_valid_embeddings[-cur_data_num:]
                    cur_valid_embeddings = cur_valid_embeddings[-cur_data_num:]
                    embedding_num = cur_embeddings.shape[0]

                    if pre_pairwise_dist is None:
                        pairwise_distance = get_pairwise_distance(total_data)
                        pre_pairwise_dist = pairwise_distance
                    else:
                        new2cur_dist = cdist(new_data, total_data)
                        pairwise_distance = np.concatenate([pre_pairwise_dist, new2cur_dist[:, :-new_data_num]], axis=0)
                        pairwise_distance = np.concatenate([pairwise_distance, new2cur_dist.T], axis=1)
                        pre_pairwise_dist = pairwise_distance

                    pre_time_step = cur_time_step
                    data_idx += new_data_num
                    eval_time += 1
                    output = ""
                    if method == "Xtreaming" and eval_time % 2 != 1:
                        for ii, metric in enumerate(metric_list):
                            metric_records[ii].append(metric_records[ii][-1])
                        continue

                    pre_embedding_pdist = get_pairwise_distance(cur_embeddings)
                    pre_e_num = pre_embedding_pdist.shape[0]

                    eval_pdist = pairwise_distance[:embedding_num, :][:, :embedding_num]
                    knn_indices, knn_dists = cal_knn(eval_pdist)

                    metric_tool = Metric(dataset, total_data[:embedding_num], total_label[:embedding_num], knn_indices,
                                         knn_dists, eval_pdist, k=eval_k)
                    metric_tool.low_dis_matrix = pre_embedding_pdist

                    shepard_good = metric_tool.metric_shepard_diagram_correlation(cur_embeddings)
                    demap = metric_tool.metric_demap(cur_embeddings)
                    metric_records[0].append(shepard_good)
                    metric_records[1].append(demap)
                    output += "Shepard: %.4f DEMaP: %.4f" % (shepard_good, demap)
                    print(jtem, output)

                metric_records = np.array(metric_records)
                avg_metric_records = np.mean(metric_records, axis=1)
                np.save(os.path.join(item_dir, "add_metric_records.npy"), metric_records)
                dataset_res.append(avg_metric_records)
                out_put = ""
                indices = np.arange(len(metric_records[0]))
                for ii, metric in enumerate(metric_list):
                    simple_draw(indices, metric_records[ii], "%s-%.4f" % (metric, avg_metric_records[ii]),
                                os.path.join(metric_image_dir, "{}.jpg".format(metric)))
                    out_put += " %s: %.4f" % (metric, avg_metric_records[ii])

                print(out_put)
                metric_file = open(os.path.join(item_dir, "add_metric_res.txt"), "w")
                metric_file.write(out_put)
                metric_file.close()

            dataset_res_npy = np.array(dataset_res)
            avg_res = np.mean(dataset_res_npy, axis=0)
            dataset_res_npy = np.concatenate([dataset_res_npy, avg_res[np.newaxis, :]], axis=0).astype(float)
            for ii, metric in enumerate(metric_list):
                res_dict[metric] = dataset_res_npy[:, ii]

            xlsx_save_path = os.path.join(dataset_dir, "res.xlsx")
            df = pd.DataFrame(res_dict)
            df.to_excel(xlsx_save_path)

            for k, item in enumerate(valid_metric_indices):
                total_res_data[k, j, i] = avg_res[item]

            j += 1

    for i, metric in enumerate(metric_list):
        res_dict = {"Dataset": dataset_list}
        for j, method in enumerate(method_list):
            res_dict[method] = total_res_data[i, :, j]

        df = pd.DataFrame(res_dict)
        df.to_excel(os.path.join(save_dir, "{}.xlsx".format(metric)))