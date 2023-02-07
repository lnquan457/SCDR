import os

import h5py
import pandas as pd
import h5py as hp
import numpy as np


if __name__ == '__main__':
    data_dir = r"D:\Projects\流数据\Data\new\datasets\1 Datasets_Healthy_Older_People\total"
    save_dir = r"D:\Projects\流数据\Data\new"

    total_data = []
    total_labels = []

    for item in os.listdir(data_dir):
        if str(item).endswith(".txt"):
            continue
        data_path = os.path.join(data_dir, item)
        df = open(data_path, "r")
        for line in df.readlines():
            cur_data = np.fromstring(line, dtype=float, sep=",")
            cur_data_info = cur_data[1:]
            cur_label = cur_data[-1].astype(int)

            total_data.append(cur_data_info)
            total_labels.append(cur_label)
    #
    with h5py.File(os.path.join(save_dir, "activity_rec.h5"), "w") as hf:
        hf['x'] = np.array(total_data, dtype=float)
        hf['y'] = np.array(total_labels, dtype=int)