#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
from utils.parser import *
from utils.constant_pool import *
from model.dr_models.ModelSets import *
import argparse
from experiments.scdr_trainer import SCDRTrainer

device = "cuda:0"
log_path = "logs/logs.txt"


def scdr_pipeline(config_path):
    cfg.merge_from_file(config_path)

    result_save_dir = ConfigInfo.RESULT_SAVE_DIR.format(cfg.model_params.name, cfg.exp_params.n_neighbors,
                                                        cfg.model_params.latent_dim)
    clr_model = MODELS[cfg.model_params.name](cfg, device=device)

    experimenter = SCDRTrainer(clr_model, cfg.exp_params.dataset, config_path, cfg, result_save_dir,
                               device=device, log_path=log_path)

    # experimenter.normal_start()
    # ckpt_path = r"H:\Projects\流数据\Code\SCDR\results\NX_CDR\n10_d2\isolet_subset_20220412_17h20m04s\NX_CDR_400.pth.tar"
    ckpt_path = r"H:\Projects\流数据\Code\SCDR\results\CDR\n10_d2\isolet_subset_20220509_14h25m20s\CDR_400.pth.tar"
    # experimenter.load_weights_start(ckpt_path)


def scdr_custom_indices_pipeline(config_path):
    cfg.merge_from_file(config_path)

    result_save_dir = ConfigInfo.RESULT_SAVE_DIR.format(cfg.model_params.name, cfg.exp_params.n_neighbors,
                                                        cfg.model_params.latent_dim)
    # 创建CDR模型
    clr_model = MODELS[cfg.model_params.name](cfg, device=device)

    custom_indices_path = os.path.join(ConfigInfo.CUSTOM_INDICES_DIR, "{}.npy".format(cfg.exp_params.dataset))
    custom_indices = np.load(custom_indices_path, allow_pickle=True)

    experimenter = SCDRTrainer(None, custom_indices, clr_model, cfg.exp_params.dataset, config_path, cfg,
                               result_save_dir, device=device, log_path=log_path)

    # experimenter.normal_start()
    # ckpt_path = r"H:\Projects\流数据\Code\SCDR\results\NX_CDR\n10_d2\isolet_subset_20220412_17h09m13s\NX_CDR_400.pth.tar"
    ckpt_path = r"H:\Projects\流数据\Code\SCDR\results\CDR\n10_d2\isolet_subset_20220509_14h25m20s\CDR_400.pth.tar"
    experimenter.load_weights_start(ckpt_path)


PIPE_LINES = {
    "NX_CDR": scdr_pipeline,
    # "NX_CDR": scdr_custom_indices_pipeline,
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default="NX_CDR", choices=["CDR", "NX_CDR"])
    parser.add_argument("--configs", type=str, default="configs/clr/SCDR.yaml", help="configuration file path")

    parser.add_argument("-Xmx", type=str, default="102400m")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config()

    PIPE_LINES[args.method](args.configs)
