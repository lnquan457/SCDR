# coding=utf-8
import os.path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP

from model.scdr.dependencies.experiment import position_vis

if __name__ == '__main__':
    data_dir = r"D:\Projects\流数据\Code\SCDR\results\excel_res\20230217_15h30m42s"
    save_dir = r"D:\Projects\流数据\Code\SCDR\results\excel_res\per_metrics_0217"
    total_res = np.empty((3, 7, 5))
    method_list = ["sPCA", "Xtreaming", "SIsomap++", "INE", "SCDR"]
    metric_indices = [-3, -2, -1]
    metric_list = ["Trust", "Neighbor Hit", "KA(10)", "Position Change", "Single Process Time", "Total Process Time", "Delay Time"]
    dataset_list = ["arem", "basketball", "HAR_2", "mnist_fla", "sat", "shuttle", "usps"]

    for i, method in enumerate(method_list):
        print("=================", method)
        method_dir = os.path.join(data_dir, method)
        for j, dataset in enumerate(dataset_list):
            print("*****************", dataset)
            if method == "sPCA" and dataset == "mnist_fla":
                data = [0 for i in range(4)]
                data = np.array([data])
            else:
                file_path = os.path.join(method_dir, "{}.xlsx".format(dataset))
                df = pd.read_excel(file_path)
                data = np.array(df.values)
            for idx, k in enumerate(metric_indices):
                total_res[idx, j, i] = data[-1][k]

    for i, metric in enumerate(metric_list[-3:]):
        res_dict = {"Dataset": dataset_list}
        for j, method in enumerate(method_list):
            res_dict[method] = total_res[i, :, j]

        df = pd.DataFrame(res_dict)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df.to_excel(os.path.join(save_dir, "{}.xlsx".format(metric)))

