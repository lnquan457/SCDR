import os

import numpy as np
import pandas as pd

if __name__ == '__main__':
    data_dir = r"D:\Projects\流数据\Evaluation\原始数据\ND\0224"
    save_dir = r"D:\Projects\流数据\Evaluation\表格数据\ND\0224"
    total_res = np.zeros((5, 10, 5))
    metric_name_list = ["Trust", "Continuity", "Neighbor Hit", "kNN Acc", "Position Change"]
    method_list = ["sPCA", "Xtreaming", "SIsomap++", "INE", "SCDR"]
    dataset_list = []

    for i, method in enumerate(method_list):
        method_dir = os.path.join(data_dir, method)
        for j, dataset in enumerate(os.listdir(method_dir)):
            if i == 0:
                dataset_list.append(dataset)
            file_path = os.path.join(method_dir, dataset, "res.xlsx")
            df = pd.read_excel(file_path)
            metrics = df.values[2, 1:]

            total_res[i, j] = metrics

    for i, item in enumerate(metric_name_list):
        res_dict = {"Dataset": dataset_list}
        for j, jtem in enumerate(method_list):
            res_dict[jtem] = total_res[j, :, i]
        df = pd.DataFrame(res_dict)
        df.to_excel(os.path.join(save_dir, "{}.xlsx".format(item)))


