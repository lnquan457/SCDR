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
    indices_dir = r"D:\Projects\流数据\Data\new\indices_seq"
    dataset_list = ["arem", "basketball", "shuttle", "HAR_2", "mnist_fla"]
    situation = "PD"
    for dataset in dataset_list:
        initial_indices, after_indices = np.load(os.path.join(indices_dir, "{}_{}_new.npy".format(dataset, situation)),
                                                 allow_pickle=True)
        print(dataset, len(after_indices))