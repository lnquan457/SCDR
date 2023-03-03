# coding=utf-8
import os.path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP

from model.scdr.dependencies.experiment import position_vis
from utils.metrics_tool import Metric
from utils.nn_utils import compute_knn_graph, get_pairwise_distance

if __name__ == '__main__':
    dataset_name = "HAR_2"
    eval_k = 10
    num = 5000
    with h5py.File(r"D:\Projects\流数据\Data\H5 Data\{}.h5".format(dataset_name), "r") as hf:
        x = np.array(hf['x'], dtype=float)
        y = np.array(hf['y'], dtype=int)

        # x = x[:num]
        # y = y[:num]

        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(x)
        knn_indices, knn_dists = compute_knn_graph(x, None, eval_k, None, accelerate=False)
        pairwise_distance = get_pairwise_distance(x, pairwise_distance_cache_path=None, preload=False)
        metric_tool = Metric(dataset_name, x, y, knn_indices, knn_dists, pairwise_distance,
                             k=eval_k)
        faithful_results = metric_tool.cal_simplified_metrics(eval_k, embeddings, knn_k=eval_k)
        print(faithful_results)
