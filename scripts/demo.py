# coding=utf-8
import os.path
import random

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
from model.scdr.dependencies.experiment import position_vis
from utils.metrics_tool import Metric
from utils.nn_utils import compute_knn_graph, get_pairwise_distance

if __name__ == '__main__':
    data_dir = r"D:\Projects\流数据\Evaluation\过程数据\结果\PD\Projection Change"
    save_dir = r"D:\Projects\流数据\Evaluation\过程数据\结果\PD\Projection Change\gather_data"
    data_list = ["arem", "HAR_2", "shuttle", "mnist_fla", "basketball"]
    method_list = ["sPCA", "Xtreaming", "SCDR"]
    nums = 5000
    gather_times = 100

    range_list = [0, 0.001, 0.1, 1]

    for dataset in data_list:
        data_file_path = os.path.join(data_dir, "{}.xlsx".format(dataset))

        df = pd.read_excel(data_file_path)
        values = np.array(df.values[:-1, 1:])
        selected_indices = np.linspace(0, values.shape[0] - 1, nums).astype(int)
        # values = values[selected_indices, :]
        gather_num = values.shape[0] // gather_times
        new_res_dict = {}
        for i, method in enumerate(method_list):
            print("=====================method:", method)
            cur_val = values[:, i]
            for j in range(1, len(range_list)):
                mask = values[:, i] <= range_list[j]
                mask[values[:, i] < range_list[j-1]] = False
                print("{} ~ {}: {}".format(range_list[j-1], range_list[j], np.sum(mask)))

            cur = []
            idx = 0
            while idx < len(cur_val):
                cur.append(np.mean(cur_val[idx:idx+gather_num]))
                idx += gather_num
            new_res_dict[method] = cur

        new_res_dict["process"] = np.linspace(0, 100, len(new_res_dict["sPCA"]))
        new_df = pd.DataFrame(new_res_dict)
        new_df.to_excel(os.path.join(save_dir, "{}.xlsx".format(dataset)))

        # break
