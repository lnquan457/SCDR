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


if __name__ == '__main__':
    res_dir = r"D:\Projects\流数据\Evaluation\原始数据\PD\0222"
    save_dir = r"D:\Projects\流数据\Evaluation\表格数据\PD\0222"
    data_dir = r"D:\Projects\流数据\Data\H5 Data"
    indices_dir = r"D:\Projects\流数据\Data\new\indices_seq"
    eval_k = 10
    window_size = 5000
    valid_metric_indices = [0, 1, 2, 3, 4]
    total_res_data = np.zeros((5, 10, 2))
    xtreaming_buffer_size = 200
    eval_step = 100
    situation = "PD"
    method_list = ["sPCA", "Xtreaming", "SIsomap++", "INE", "SCDR"]
    # method_list = ["sPCA", "Xtreaming", "INE", "SCDR", "SIsomap++"]
    # method_list = ["sPCA"]
    metric_list = ["Position Change", "Big Position Change"]
    dataset_list = ["mnist_fla"]
    # dataset_list = ["arem", "basketball", "HAR_2", "mnist_fla", "sat", "shuttle", "usps", "Anuran Calls_8c",
    #                 "electric_devices", "texture"]
    # dataset_list = ["Anuran Calls_8c", "electric_devices", "texture"]

    for i, method in enumerate(method_list):
        print("Method:", method)
        method_dir = os.path.join(res_dir, method)
        j = 0
        for dataset in dataset_list:
            print("==========Dataset:", dataset)

            dataset_dir = os.path.join(method_dir, dataset)
            dataset_res = np.zeros((2, 2))
            initial_indices, after_indices = np.load(os.path.join(indices_dir, "{}_{}.npy".format(dataset, situation)),
                                                     allow_pickle=True)

            for k, item in enumerate(os.listdir(dataset_dir)):
                if item.endswith("xlsx"):
                    k -= 1
                    continue
                item_dir = os.path.join(dataset_dir, item)
                eval_embedding_dir = os.path.join(item_dir, "eval_embeddings")

                metric_records = []
                big_metric_records = []
                metric_image_dir = os.path.join(item_dir, "metric_imgs")
                check_path_exist(metric_image_dir)
                sta_time_step = 1
                pre_time_step = sta_time_step
                pre_embeddings = None
                embedding_num = 0

                while True:
                    sta_time_step += eval_step
                    jtem = "t{}.npy".format(sta_time_step - 1)
                    if not os.path.exists(os.path.join(eval_embedding_dir, jtem)):
                        break
                    cur_time_step, cur_embeddings, pre_valid_embeddings, cur_valid_embeddings = \
                        np.load(os.path.join(eval_embedding_dir, jtem), allow_pickle=True)

                    new_data_num = cur_time_step - pre_time_step

                    if pre_embeddings is None:
                        pre_embeddings = cur_embeddings
                        big_step_res = 0
                        out_num = 0
                    else:
                        out_num = max(0, pre_embeddings.shape[0] - window_size + new_data_num)
                        if method == "SCDR" and out_num > 0:
                            out_num -= 2
                        tt_pre_embeddings = pre_embeddings[out_num:]
                        tt_cur_embeddings = cur_embeddings[:tt_pre_embeddings.shape[0]]
                        big_step_res = cal_global_position_change(tt_cur_embeddings, tt_pre_embeddings)
                        big_metric_records.append(big_step_res)
                        pre_embeddings = cur_embeddings

                    if out_num > 0 and method != "SCDR":
                        pre_valid_embeddings = pre_valid_embeddings[1:]
                        cur_valid_embeddings = cur_valid_embeddings[:-1]

                    if len(initial_indices) > window_size and cur_time_step == 100 and method != "SCDR":
                        cur_valid_embeddings = cur_valid_embeddings[:-1]
                        pre_valid_embeddings = pre_valid_embeddings[1:]

                    position_change = cal_global_position_change(cur_valid_embeddings, pre_valid_embeddings)
                    metric_records.append(position_change)
                    pre_time_step = cur_time_step

                    # out_put = "================ PC: %.4f Big PC: %.4f" % (position_change, big_step_res)
                    # print(jtem, out_put)

                avg_metric_records = np.mean(metric_records)
                avg_big_metric_records = np.mean(big_metric_records)
                np.save(os.path.join(item_dir, "ts_metric_records.npy"), [metric_records, big_metric_records])

                out_put = " PC: %.4f Big PC: %.4f" % (avg_metric_records, avg_big_metric_records)
                print("Final Average **************************************", out_put)
                indices = np.arange(len(metric_records))
                simple_draw(indices, metric_records, "Position Change-%.4f" % avg_metric_records,
                            os.path.join(metric_image_dir, "Position Change.jpg"))
                indices = np.arange(len(big_metric_records))
                simple_draw(indices, big_metric_records, "Position Change-%.4f" % avg_big_metric_records,
                            os.path.join(metric_image_dir, "Big Position Change.jpg"))
                dataset_res[k] = [avg_metric_records, avg_big_metric_records]

            avg_res = np.mean(dataset_res, axis=0)
            total_res_data[i, j] = avg_res
            dataset_res_npy = np.concatenate([dataset_res, avg_res[np.newaxis, :]], axis=0).astype(float)
            res_dict = {}
            for ii, metric in enumerate(metric_list):
                res_dict[metric] = dataset_res_npy[:, ii]

            xlsx_save_path = os.path.join(dataset_dir, "pc_res.xlsx")
            df = pd.DataFrame(res_dict)
            df.to_excel(xlsx_save_path)
            j += 1

    for i, metric in enumerate(metric_list):
        res_dict = {"Dataset": dataset_list}
        for j, method in enumerate(method_list):
            res_dict[method] = total_res_data[j, :, i]

        df = pd.DataFrame(res_dict)
        df.to_excel(os.path.join(save_dir, "{}_tmp.xlsx").format(metric))