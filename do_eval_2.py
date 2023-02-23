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
    res_dir = r"D:\Projects\流数据\Code\SCDR\results\eval_res\0215\raw data"
    data_dir = r"D:\Projects\流数据\Data\H5 Data"
    indices_dir = r"D:\Projects\流数据\Data\new\indices_seq"
    save_dir = r"D:\Projects\流数据\Code\SCDR\results\eval_res\0215\metric data"
    eval_k = 10
    window_size = 5000
    valid_metric_indices = [0, 1, 2, 3, 4]
    total_res_data = np.zeros((5, 7, 5))
    # method_list = ["sPCA", "Xtreaming", "SIsomap++", "INE", "Ours"]
    method_list = ["SIsomap++",]
    metric_list = ["Trust", "Continuity", "Neighbor Hit", "KA(10)", "Position Change"]
    dataset_list = ["arem", "basketball", "HAR_2", "mnist_fla", "sat", "shuttle", "usps"]

    for i, method in enumerate(method_list):
        print("Method:", method)
        method_dir = os.path.join(res_dir, method)
        j = 0
        for dataset in dataset_list:
            print("==========Dataset:", dataset)
            if method == "sPCA" and dataset == "mnist_fla":
                for k, item in enumerate(valid_metric_indices):
                    total_res_data[k, j, i] = 0
                continue

            dataset_dir = os.path.join(method_dir, dataset)
            with h5py.File(os.path.join(data_dir, "{}.h5".format(dataset)), "r") as hf:
                x = np.array(hf['x'], dtype=float)
                y = np.array(hf['y'], dtype=int)

            initial_indices, after_indices = np.load(os.path.join(indices_dir, "{}_FD.npy".format(dataset)), allow_pickle=True)
            initial_num = window_size
            initial_data = x[initial_indices]
            initial_label = y[initial_indices]

            stream_data = x[after_indices]
            stream_label = y[after_indices]
            total_raw_data = np.concatenate([initial_data, stream_data], axis=0)
            total_raw_label = np.concatenate([initial_label, stream_label])

            pre_pairwise_dist = get_pairwise_distance(initial_data[-window_size:], pairwise_distance_cache_path=None, preload=False)
            pre_embedding_pdist = None

            dataset_res = []
            res_dict = {}
            for k, item in enumerate(os.listdir(dataset_dir)):
                if item.endswith("xlsx"):
                    continue
                item_dir = os.path.join(dataset_dir, item)
                eval_embedding_dir = os.path.join(item_dir, "eval_embeddings")

                metric_records = [[] for i in range(len(METRIC_NAMES) - 1)]
                metric_image_dir = os.path.join(item_dir, "metric_imgs")
                check_path_exist(metric_image_dir)
                sta_t = 0
                eval_num = 0
                pre_data_num = initial_num
                data_idx = len(initial_indices)
                pre_time_step = 0

                while True:
                    sta_t += 100
                    jtem = "t{}.npy".format(sta_t)
                    if not os.path.exists(os.path.join(eval_embedding_dir, jtem)):
                        break
                    print(jtem)
                    cur_time_step, cur_embeddings, pre_valid_embeddings, cur_valid_embeddings = \
                        np.load(os.path.join(eval_embedding_dir, jtem), allow_pickle=True)
                    cur_embeddings = cur_embeddings[-window_size:]
                    pre_valid_embeddings = pre_valid_embeddings[-window_size:]
                    cur_valid_embeddings = cur_valid_embeddings[-window_size:]
                    data_idx += cur_time_step

                    new_data_num = cur_time_step - pre_time_step
                    out_num = max(0, pre_data_num + new_data_num - window_size)

                    if out_num > 0:
                        pre_pairwise_dist = pre_pairwise_dist[out_num:, ][:, out_num:]

                    sta = time.time()
                    pre_data_num = pre_pairwise_dist.shape[0]
                    new_data = total_raw_data[data_idx-new_data_num:data_idx]
                    cur_data = total_raw_data[:data_idx][-cur_embeddings.shape[0]:]
                    new_embeddings = cur_embeddings[-new_data_num:]

                    cur_label = total_raw_label[:data_idx][-cur_embeddings.shape[0]:]
                    new2cur_dist = cdist(new_data, cur_data)

                    pairwise_distance = np.concatenate([pre_pairwise_dist, new2cur_dist[:, :pre_data_num]], axis=0)
                    pairwise_distance = np.concatenate([pairwise_distance, new2cur_dist.T], axis=1)
                    pre_pairwise_dist = pairwise_distance

                    if pre_embedding_pdist is None:
                        pre_embedding_pdist = get_pairwise_distance(cur_embeddings)
                    elif out_num > 0:
                        pre_embedding_pdist = pre_embedding_pdist[out_num:, ][:, out_num:]

                        new2cur_e_dist = cdist(new_embeddings, cur_embeddings)
                        embedding_pdist = np.concatenate([pre_embedding_pdist, new2cur_e_dist[:, :pre_data_num]], axis=0)
                        embedding_pdist = np.concatenate([embedding_pdist, new2cur_e_dist.T], axis=1)
                        pre_embedding_pdist = embedding_pdist

                    # pairwise_distance = get_pairwise_distance(cur_data, pairwise_distance_cache_path=None,
                    #                                           preload=False)
                    knn_indices, knn_dists = cal_knn(pairwise_distance)
                    pre_time_step = cur_time_step
                    pre_data_num += new_data_num

                    metric_tool = Metric(dataset, cur_data, cur_label, knn_indices, knn_dists,
                                         pairwise_distance, k=eval_k)
                    metric_tool.low_dis_matrix = pre_embedding_pdist

                    faithful_results = metric_tool.cal_simplified_metrics(eval_k, cur_embeddings, knn_k=eval_k)[:4]
                    for ii, metric in enumerate(faithful_results):
                        metric_records[ii].append(metric)

                    position_change = cal_global_position_change(cur_valid_embeddings, pre_valid_embeddings)
                    metric_records[-1].append(position_change)

                metric_records = np.array(metric_records)
                avg_metric_records = np.mean(metric_records, axis=1)
                dataset_res.append(avg_metric_records)
                out_put = ""
                indices = np.arange(len(metric_records[0]))
                for ii, metric in enumerate(METRIC_NAMES[:4]):
                    simple_draw(indices, metric_records[ii], "%s-%.4f" % (metric, avg_metric_records[ii]),
                                os.path.join(metric_image_dir, "{}.jpg".format(metric)))
                    out_put += " %s: %.4f" % (metric, avg_metric_records[ii])

                simple_draw(indices, metric_records[-1], "Position Change-%.4f" % avg_metric_records[-1],
                            os.path.join(metric_image_dir, "Position Change.jpg"))
                out_put += " Position Change: %.4f\n" % avg_metric_records[-1]
                print(out_put)
                metric_file = open(os.path.join(item_dir, "metric_res.txt"), "w")
                metric_file.write(out_put)
                metric_file.close()

            dataset_res_npy = np.array(dataset_res)
            avg_res = np.mean(dataset_res_npy, axis=0)
            dataset_res_npy = np.concatenate([dataset_res_npy, avg_res[np.newaxis, :]], axis=0).astype(float)
            for ii, metric in enumerate(METRIC_NAMES[:4]):
                res_dict[metric] = dataset_res_npy[:, ii]

            res_dict["Position Change"] = dataset_res_npy[:, -1]
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