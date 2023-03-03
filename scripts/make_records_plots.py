import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator

from utils.constant_pool import FINAL_DATASET_LIST

if __name__ == '__main__':
    data_dir = r"D:\Projects\流数据\Evaluation\过程数据\原始数据\投影质量"
    save_dir = r"D:\Projects\流数据\Evaluation\过程数据\结果"
    dataset_list = FINAL_DATASET_LIST
    method_list = ["sPCA", "Xtreaming", "SIsomap++", "INE", "SCDR"]
    metric_list = ["Trust", "Continuity", "Neighbor Hit", "kNN-CA", "Big Projection Change"]
    # metric_list = ["Big Projection Change"]
    target_metric_indices = [0, 1, 2, 3, 4]
    # target_metric_indices = [0]
    # metric * dataset * method
    total_data = np.empty((len(target_metric_indices), len(FINAL_DATASET_LIST), 5), dtype=object)

    for i, dataset in enumerate(dataset_list):
        print("Processing:", dataset)
        dataset_dir = os.path.join(data_dir, dataset)
        for j, method in enumerate(method_list):
            metric_file_path = os.path.join(dataset_dir, "{}.npy".format(method))
            metric_data = np.load(metric_file_path, allow_pickle=True)

            for k in target_metric_indices:
                total_data[k, i, j] = metric_data[k]

    for i in target_metric_indices:
        for j, dataset in enumerate(dataset_list):
            res_dict = {}
            plt.figure()
            ax = plt.gca()
            indices = None
            for k, method in enumerate(method_list):
                data = list(total_data[i, j, k])

                if indices is None:
                    indices = np.linspace(0, 100, len(data))
                    indices = ["{}%".format(item) for item in indices]
                    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(data), decimals=0))
                    ax.xaxis.set_major_locator(MultipleLocator(len(data)//4))

                data.append(np.mean(total_data[i, j, k]))
                res_dict[method] = data

                plt.plot(data[:-1], label=method)

            df = pd.DataFrame(res_dict)
            cur_save_dir = os.path.join(save_dir, metric_list[i])
            if not os.path.exists(cur_save_dir):
                os.makedirs(cur_save_dir)
            df.to_excel(os.path.join(cur_save_dir, "{}.xlsx".format(dataset)))
            img_save_dir = os.path.join(cur_save_dir, "imgs")
            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir)
            # plt.ylabel(metric_list[i])
            # plt.legend()
            plt.savefig(os.path.join(img_save_dir, "{}.jpg".format(dataset)), dpi=400, bbox_inches='tight', pad_inches=0.1)
