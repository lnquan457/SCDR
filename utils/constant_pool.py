#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from multiprocessing import Queue

METRIC_NAMES = ["Trust", "Continuity", 'Neighbor Hit', 'KA(10)', 'KA(5)', 'KA(1)', 'Sim Fake', 'Dissim Lost', 'SC',
                'DSC', 'GON']
ALL_DATASETS = ["animals", "banknote", "Anuran Calls_8c", "cifar10_ex_10000", "cnae9", "dogs_cats_10000_idx",
                "fish", "food", "har", "isolet_subset", "mnist_r_10000",
                "ml binary", "pendigits", "retina_r_10000", "satimage", "stanford dogs_subset",
                "texture", "usps_clear", "wethers", "wifi"]
DATASET_MAP = {"stanford dogs": "stanford dogs_subset", "animals": "animals_clear", "pendigits": "pendigits_clear"}

SIPCA = "siPCA"
ATSNE = "atSNE"
XTREAMING = "Xtreaming"
SCDR = "SCDR"
RTSCDR = "RTSCDR"

# STREAM_METHOD_LIST = [SIPCA, XTREAMING, ATSNE, SCDR]
# STREAM_METHOD_LIST = [ATSNE, SCDR]
# STREAM_METHOD_LIST = [XTREAMING]
# STREAM_METHOD_LIST = [SCDR, ATSNE, XTREAMING, SIPCA]
STREAM_METHOD_LIST = [SCDR]


def name_map(name):
    name = name.lower()
    if name not in DATASET_MAP.keys():
        return name
    return DATASET_MAP[name]


class ProjectSettings:
    LABEL_COLORS = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red', 4: 'blueviolet', 5: 'maroon', 6: 'deeppink',
                    7: 'greenyellow', 8: 'olive', 9: 'cyan', 10: 'yellow', 11: 'purple'}
    QUANTITATIVE_METRIC_NAME = ["Trust", "Continuity", 'Neighbor Hit', 'KNN ACC', 'Sim Fake', 'Dissim Lost', 'Acc',
                                'NMI', 'SC', 'DSC', 'GON']


class ConfigInfo:
    # method_name, dataset_name, method_name+dataset_name.ckpt
    MODEL_CONFIG_PATH = "./configs/{}.yaml"
    RESULT_SAVE_DIR = r".\results\{}\n{}_d{}"
    # 单机下的数据缓存目录
    NEIGHBORS_CACHE_DIR = r"..\..\Data\knn\{}_k{}.npy"
    PAIRWISE_DISTANCE_DIR = r"..\..\Data\pairwise\{}.npy"
    DATASET_CACHE_DIR = r"..\..\Data\H5 Data"
    CUSTOM_INDICES_DIR = "../../../Data/indices"


class ComponentInfo:
    # 归一化方法
    UMAP_NORMALIZE = "umap"
    TSNE_NORMALIZE = "tsne"
    NONE_NORMALIZE = "none"


class ModelName:
    # VAEs
    LinearVAE = "LinearVAE"
    ConvVAE = "ConvVAE"

    # UMAPs
    Par_UMAP = "Par_UMAP"

    # CDRs
    SimCLR = "SimCLR"
    NNCLR = "NNCLR"
