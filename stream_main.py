import argparse

import numpy as np
import os

from experiments.scdr_trainer import SCDRTrainer
from experiments.streaming_experiment import StreamingEx
from model.dr_models.ModelSets import MODELS
from utils.constant_pool import ConfigInfo, SIPCA, ATSNE, XTREAMING, SCDR, STREAM_METHOD_LIST
from utils.parser import get_config

device = "cuda:0"
log_path = "logs/logs.txt"


def stream_rate_ex():
    buffer_size = 100
    # rate_list = [2, int(buffer_size * 0.5), buffer_size, buffer_size * 2]
    rate_list = [2]
    # rate_list = [100]
    # rate_list = [buffer_size * 2]
    # rate_list = [int(buffer_size * 0.5)]

    global result_save_dir, cfg

    for i, item in enumerate(rate_list):
        for j, m in enumerate(STREAM_METHOD_LIST):
            print("正在处理 r = {} m = {}".format(item, m))
            cfg = get_config()
            cfg.merge_from_file(ConfigInfo.MODEL_CONFIG_PATH.format(m))
            cfg.exp_params.stream_rate = item
            if item != 2:
                cfg.exp_params.vis_iter = 1
            args.method = m
            if m == SCDR:
                m = m + "_r1.0"
            result_save_dir = "results/初步实验/stream_rate/single cluster/r{}/{}".format(item, m)
            # result_save_dir = "results/初步实验/stream_rate/multi cluster/r{}/{}".format(item, m)

            custom_indices_path = r"H:\Projects\流数据\Data\indices\single_cluster\{}.npy".format(cfg.exp_params.dataset)
            # custom_indices_path = r"H:\Projects\流数据\Data\indices\multi_cluster\{}_3_3_3_2.npy".format(
            #     cfg.exp_params.dataset)
            custom_indices_training(custom_indices_path)


def cluster_composite_ex():
    global cfg, result_save_dir
    stream_rate = 2
    for i, m in enumerate(STREAM_METHOD_LIST):
        cfg = get_config()
        cfg.merge_from_file(ConfigInfo.MODEL_CONFIG_PATH.format(m))
        cfg.exp_params.stream_rate = stream_rate
        args.method = m
        if m == SCDR:
            m = m + "_r1.0"
        result_save_dir = "results/初步实验/cluster composite/single cluster/recurring/{}".format(m)
        # result_save_dir = "results/初步实验/cluster composite/multi cluster/recurring/{}".format(m)
        # result_save_dir = "results/初步实验/cluster composite/identical distribution/{}".format(m)
        custom_indices_path = r"H:\Projects\流数据\Data\indices\single_cluster_recur\{}_recur2.npy".format(
            cfg.exp_params.dataset)
        # custom_indices_path = r"H:\Projects\流数据\Data\indices\multi_cluster_recur\{}_3_3_3_2_recur2.npy".format(
        #     cfg.exp_params.dataset)
        # custom_indices_path = r"H:\Projects\流数据\Data\indices\multi_cluster_recur\{}_11_recur5.npy".format(
        #     cfg.exp_params.dataset)
        custom_indices_training(custom_indices_path)


def start(ex):
    if args.method == ATSNE:
        # ==============1. at-SNE model=====================
        ex.start_atSNE(cfg.method_params.perplexity, finetune_iter=cfg.method_params.finetune_iter,
                       n_iter=cfg.method_params.initial_train_iter)
    elif args.method == SIPCA:
        # ==============2. siPCA model=====================
        ex.start_siPCA(cfg.method_params.forgetting_factor)
    elif args.method == XTREAMING:
        # ==============3. Xtreaming model=====================
        ex.start_xtreaming(cfg.method_params.buffer_size, cfg.method_params.eta)
    elif args.method == SCDR:
        # ==============4. SCDR model=====================
        cdr_model = MODELS[cfg.method_params.method](cfg, device=device)

        model_trainer = SCDRTrainer(cdr_model, cfg.exp_params.dataset, cfg_path, cfg, result_save_dir,
                                    device=device, log_path=log_path)
        # ckpt_path = r"results\SCDR\n10\isolet_subset\20220512_15h47m54s\initial\CDR_200.pth.tar"
        # ckpt_path = r"results\SCDR\n10\isolet_subset\20220512_19h34m47s\initial\CDR_400.pth.tar"
        ckpt_path = None
        ex.start_scdr(cfg.method_params.n_neighbors, cfg.method_params.buffer_size, model_trainer,
                      cfg.method_params.initial_train_epoch, cfg.method_params.finetune_epoch, ckpt_path)
    else:
        raise RuntimeError("Non-supported method! please ensure param 'method' is one of 'atSNE/siPCA/Xtreaming/SCDR'!")


def random_training():
    # labels = np.array(h5py.File(os.path.join(ConfigInfo.DATASET_CACHE_DIR, "{}.h5".format(args.dataset)), "r")["y"])
    # unique_labels = np.unique(labels)
    # np.random.shuffle(unique_labels)
    # initial_cls_ratio = 0.6
    # initial_labels = unique_labels[:int(initial_cls_ratio * len(unique_labels))]
    # print("initial_labels", initial_labels)
    initial_labels = [15, 24, 25]

    ex = StreamingEx(initial_labels, cfg, None, result_save_dir)

    start(ex)


def custom_indices_training(custom_indices_path):
    # custom_indices_path = os.path.join(ConfigInfo.CUSTOM_INDICES_DIR, "{}.npy".format(args.dataset))
    custom_indices = np.load(custom_indices_path, allow_pickle=True)

    ex = StreamingEx(None, cfg, custom_indices, result_save_dir)

    start(ex)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default=XTREAMING, choices=[ATSNE, SIPCA, XTREAMING, SCDR])
    parser.add_argument("-Xmx", type=str, default="102400m")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg_path = ConfigInfo.MODEL_CONFIG_PATH.format(args.method)
    cfg = get_config()
    cfg.merge_from_file(cfg_path)
    if args.method == ATSNE:
        result_save_dir = "results/{}/n{}/".format(args.method, cfg.method_params.perplexity)
    elif args.method == SCDR:
        result_save_dir = "results/{}/n{}/".format(args.method, cfg.method_params.n_neighbors)
    else:
        result_save_dir = "results/{}/".format(args.method)

    # random_training()
    # custom_indices_path = r"H:\Projects\流数据\Data\indices\single_cluster\{}.npy".format(cfg.exp_params.dataset)
    # custom_indices_training(custom_indices_path)
    stream_rate_ex()
    # cluster_composite_ex()