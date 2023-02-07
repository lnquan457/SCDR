import os

import h5py
import pandas as pd
import h5py as hp
import numpy as np


if __name__ == '__main__':
    data_dir = r"D:\Projects\流数据\Data\new\datasets\1 AReM"
    save_dir = r"D:\Projects\流数据\Data\new"

    total_data = None
    total_labels = None

    cls_idx = 0
    for item in os.listdir(data_dir):
        if str(item).endswith(".pdf"):
            continue
        cls_num = 0
        print("==============", item)
        for subitem in os.listdir(os.path.join(data_dir, item)):
            print("****", subitem)
            data_path = os.path.join(data_dir, item, subitem)

            df = pd.read_csv(data_path, header=None, sep=",", skiprows=5)
            # df = pd.read_csv(data_path)

            data = np.array(df, dtype=float)[:, 1:]
            cls_num += data.shape[0]
            if total_data is None:
                total_data = data
            else:
                total_data = np.concatenate([total_data, data], axis=0)

        cls_labels = [cls_idx] * cls_num
        cls_labels = np.array(cls_labels, dtype=int)
        cls_idx += 1

        if total_labels is None:
            total_labels = cls_labels
        else:
            total_labels = np.concatenate([total_labels, cls_labels])
    #
    with h5py.File(os.path.join(save_dir, "arem.h5"), "w") as hf:
        hf['x'] = total_data.astype(float)
        hf['y'] = total_labels.astype(int)