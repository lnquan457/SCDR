import os

import h5py
import pandas as pd
import h5py as hp
import numpy as np


if __name__ == '__main__':
    data_path = r"D:\Projects\流数据\Data\new\datasets\1 shuttle\shuttle.hdf5"
    save_dir = r"D:\Projects\流数据\Data\new"

    with hp.File(data_path, "r") as hf:
        data = np.array(hf['x'])
        labels = np.array(hf['y'])
        cls_names = np.unique(labels)
        int_labels = []
        for item in labels:
            # int_labels.append(list(cls_names).index(item))
            int_labels.append(int(item))

    with hp.File(os.path.join(save_dir, "shuttle.h5"), "w") as hf2:
        hf2['x'] = data
        hf2['y'] = np.array(int_labels, dtype=int)

    # df = pd.read_csv(data_path, header=None, sep=',')
    # df = pd.read_csv(data_path)
    #
    # data = np.array(df)
    #
    # labels = data[:, -1].astype(int)
    # data = data[:, 1:-1].astype(float)
    #
    # items = [10000000, 10000, 5000, 2000, 1000]
    # for i, item in enumerate(items):
    #     labels[labels < item] = i
    #
    # # for i, item in enumerate(data):
    # #     for j in [0, 2, 4]:
    # #         data[i, j] = ord(data[i, j]) - ord('a')
    #
    # # data = data[:, :-1]
    # # labels = data[:, -1]
    # # cls_names = np.unique(labels)
    # # int_labels = []
    # # for item in labels:
    # #     int_labels.append(list(cls_names).index(item))
    #
    # with h5py.File(os.path.join(save_dir, "news_popularity.h5"), "w") as hf:
    #     hf['x'] = data.astype(float)
    #     hf['y'] = np.array(labels, dtype=int)