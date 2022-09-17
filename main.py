#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

from model.dr_models.ModelSets import MODELS
from utils.common_utils import get_config

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
import torch
from utils.constant_pool import *
import argparse
import time
from model.scdr.dependencies.cdr_experiment import CDRsExperiments

device = "cuda:0"
log_path = "logs/log.txt"


def cdr_pipeline(config_path):
    # YAML配置文件路径
    cfg.merge_from_file(config_path)
    result_save_dir = ConfigInfo.RESULT_SAVE_DIR.format(method_name, cfg.method_params.n_neighbors,
                                                        cfg.exp_params.latent_dim)
    # 创建CLR模型
    clr_model = MODELS[method_name](cfg, device=device)
    experimenter = CDRsExperiments(clr_model, cfg.exp_params.dataset, cfg, result_save_dir, config_path, shuffle=True,
                                   device=device, log_path=log_path)

    sta = time.time()
    # 训练模型
    # 使用采样的小样本进行训练，采样率可以为0.5， 0.2， 0.1，然后在全部样本上进行测试
    # experimenter.subsample_train = True
    # experimenter.train_for_visualize()
    experimenter.average_metrics_test(test_time=3)


PIPE_LINES = {
    "CDR": cdr_pipeline,
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default="CDR", choices=["CDR"])
    parser.add_argument("--configs", type=str, default="configs/CDR.yaml", help="configuration file path")

    parser.add_argument("-Xmx", type=str, default="102400m")

    return parser.parse_args()


if __name__ == '__main__':
    with torch.cuda.device(0):
        args = parse_args()
        cfg = get_config()
        method_name = args.method
        PIPE_LINES[args.method](args.configs)
