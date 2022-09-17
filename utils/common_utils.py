#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import time

import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from yaml import FullLoader

from dataset.datasets import MyImageDataset, MyTextDataset
from utils.constant_pool import ProjectSettings
from utils.logger import InfoLogger


def visualize(model, data, labels, vis_title, embeddings=None, vis_save_path=None, device="cuda"):
    if embeddings is None:
        with torch.no_grad():
            z = model.acquire_latent_code(data)
            if device == "cuda":
                z = z.cpu()
        z = z.numpy()
    else:
        z = embeddings.numpy()

    x = z[:, 0]
    y = z[:, 1]

    num_classes = np.unique(labels)

    if num_classes <= 10:
        for num in num_classes:
            cur_indices = np.argwhere(labels == num)
            cur_x = x[cur_indices]
            cur_y = y[cur_indices]
            plt.scatter(cur_x, cur_y, c=ProjectSettings.LABEL_COLORS[num], s=1,
                        alpha=0.8, rasterized=True)
        plt.legend(ProjectSettings.LABEL_COLORS.keys(), markerscale=3)
    else:
        plt.scatter(x, y, c=labels, cmap="tab20", s=1, alpha=0.8, rasterized=True)

    plt.axis('equal')
    plt.title(vis_title, fontsize=20)

    if vis_save_path is not None:
        plt.savefig(vis_save_path)
    plt.show()

    return z


def draw_loss(training_loss, idx, save_path=None):
    # 画出总损失图
    plt.figure()
    plt.plot(idx, training_loss, color="blue", label="training loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def symmetry_knn_umap(knn_indices, knn_weights):
    n_sample, n_neighbor = knn_indices.shape
    rows = np.repeat(range(n_sample), n_neighbor).flatten()
    cols = knn_indices.flatten()
    weights = knn_weights.flatten()
    coo = scipy.sparse.coo_matrix(
        (weights, (rows, cols)), shape=(n_sample, n_sample)
    )
    sym_coo = (coo + coo.transpose()).tocoo()
    nn_indices, nn_weights = extract_coo(sym_coo)
    return nn_indices, nn_weights


def extract_coo(coo):
    n_sample = coo.shape[0]
    nn_indices = []
    nn_weights = []

    for i in range(n_sample):
        cur_indices = np.argwhere(coo.row == i)
        cur_indices, cur_weights = coo.col[cur_indices].flatten(), coo.data[cur_indices].flatten()
        re_indices = np.argsort(cur_weights)[::-1]
        nn_indices.append(cur_indices[re_indices])
        nn_weights.append(cur_weights[re_indices])
    return nn_indices, nn_weights


def extract_knn_from_csr(csr_graph, n_samples):
    nn_indices = []
    nn_weights = []

    for i in range(1, n_samples + 1):
        idx = csr_graph.indptr[i]
        cur_indices = csr_graph.indices[csr_graph.indptr[i - 1]:idx]
        cur_weights = csr_graph.data[csr_graph.indptr[i - 1]:idx]

        nn_indices.append(cur_indices)
        nn_weights.append(cur_weights)
    return nn_indices, nn_weights


def extract_sorted_knn_from_csr(csr_graph, n_samples, increase=True):
    nn_indices = []
    nn_data = []

    for i in range(1, n_samples + 1):
        idx = csr_graph.indptr[i]
        cur_indices = csr_graph.indices[csr_graph.indptr[i - 1]:idx]
        cur_data = csr_graph.data[csr_graph.indptr[i - 1]:idx]
        re_indices = np.argsort(cur_data)
        if not increase:
            re_indices = re_indices[::-1]
        cur_indices = cur_indices[re_indices]
        cur_data = cur_data[re_indices]

        nn_indices.append(cur_indices)
        nn_data.append(cur_data)
    return nn_indices, nn_data


def evaluate_and_log(metric_tool, embeddings, eval_k, knn_k, log_file):
    trust, continuity, neighbor_hit, knn_ac, sc, dsc = metric_tool.cal_all_metrics(eval_k, embeddings, knn_k=knn_k)
    metric_template = "Trust: %.4f Continuity: %.4f Neighbor Hit: %.4f KA: %.4f SC: %.4f DSC: %.4f"
    output = metric_template % (trust, continuity, neighbor_hit, knn_ac, sc, dsc)
    if log_file is not None:
        log_file.write(output + "\n")
    InfoLogger.info(output)
    return float(trust), float(continuity), neighbor_hit, knn_ac, sc, dsc


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert (os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read(), Loader=FullLoader))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.load(fo.read(), Loader=FullLoader))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)


DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"
DATE_ADJOIN_FORMAT = "%Y%m%d"
DATE_TIME_ADJOIN_FORMAT = "%Y%m%d_%Hh%Mm%Ss"


def time_stamp_to_date_time(time_stamp):
    time_array = time.localtime(time_stamp)
    otherStyleTime = time.strftime(DATE_TIME_FORMAT, time_array)
    return otherStyleTime


def time_stamp_to_date(time_stamp):
    time_array = time.localtime(time_stamp)
    return time.strftime(DATE_FORMAT, time_array)


def time_stamp_to_date_adjoin(time_stamp):
    time_array = time.localtime(time_stamp)
    return time.strftime(DATE_ADJOIN_FORMAT, time_array)


def time_stamp_to_date_time_adjoin(time_stamp):
    time_array = time.localtime(time_stamp)
    return time.strftime(DATE_TIME_ADJOIN_FORMAT, time_array)


def date_time2timestamp(str_time):
    # 转换成时间数组
    timeArray = time.strptime(str_time, DATE_TIME_FORMAT)
    # 转换成时间戳
    timestamp = time.mktime(timeArray)
    return timestamp


def simple_transforms():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform


def get_dataset(dataset_name, root_dir, train, transform=None, is_image=True):
    if is_image:
        dataset = MyImageDataset(dataset_name, root_dir, train, transform)
    else:
        dataset = MyTextDataset(dataset_name, root_dir, train)
    return dataset


def get_dataloaders(dataset_name, root_dir, batch_size, shuffle, transform=None, is_image=True):
    train_dataset = get_dataset(dataset_name, root_dir, True, transform, is_image)
    # test_dataset = get_dataset(dataset_name, root_dir, False, transform, is_image)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, None