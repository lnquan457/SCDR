# coding=utf-8
import os.path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP

from model.scdr.dependencies.experiment import position_vis
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = np.random.randint(1, 100, size=(100, 2))
    data_2 = np.copy(data).astype(np.float)
    cur_x_min, cur_y_min = np.min(data_2, axis=0)
    cur_x_max, cur_y_max = np.max(data_2, axis=0)
    data_2[:, 0] = (data_2[:, 0] - cur_x_min) / (cur_x_max - cur_x_min)
    data_2[:, 1] = (data_2[:, 1] - cur_y_min) / (cur_y_max - cur_y_min)

    plt.figure()
    plt.title("before")
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()

    plt.figure()
    plt.title("after")
    plt.scatter(data_2[:, 0], data_2[:, 1])
    plt.show()


