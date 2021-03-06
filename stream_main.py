import argparse
from multiprocessing.managers import BaseManager

import numpy as np
import os

from dataset.warppers import StreamingDatasetWrapper
from experiments.scdr_trainer import SCDRTrainer, SCDRTrainerProcess
from experiments.streaming_experiment import StreamingEx, StreamingExProcess
from model.dr_models.ModelSets import MODELS
from utils.constant_pool import ConfigInfo, SIPCA, ATSNE, XTREAMING, SCDR, STREAM_METHOD_LIST, RTSCDR, PAR_SCDR
from utils.parser import get_config
from utils.queue_set import ModelUpdateQueueSet

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

            custom_indices_path = r"H:\Projects\流数据\Data\indices\single_cluster\{}.npy".format(cfg.exp_params.streaming_dataset)
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
            cfg.exp_params.streaming_dataset)
        # custom_indices_path = r"H:\Projects\流数据\Data\indices\multi_cluster_recur\{}_3_3_3_2_recur2.npy".format(
        #     cfg.exp_params.dataset)
        # custom_indices_path = r"H:\Projects\流数据\Data\indices\multi_cluster_recur\{}_11_recur5.npy".format(
        #     cfg.exp_params.dataset)
        custom_indices_training(custom_indices_path)


def start(ex):
    if args.method == ATSNE:
        # ==============1. at-SNE model=====================
        ex.start_atSNE()
    elif args.method == SIPCA:
        # ==============2. siPCA model=====================
        ex.start_siPCA()
    elif args.method == XTREAMING:
        # ==============3. Xtreaming model=====================
        ex.start_xtreaming()
    elif args.method == SCDR or args.method == RTSCDR:
        # ==============4. SCDR/RTSCDR/PAR_SCDR model=====================
        cdr_model = MODELS[cfg.method_params.method](cfg, device=device)

        model_trainer = SCDRTrainer(cdr_model, cfg.exp_params.dataset, cfg_path, cfg, result_save_dir,
                                    device=device, log_path=log_path)
        if args.method == SCDR:
            ex.start_scdr(model_trainer)
        else:
            ex.start_rtscdr(model_trainer)
    elif args.method == PAR_SCDR:
        assert isinstance(ex, StreamingExProcess)
        cdr_model = MODELS[cfg.method_params.method](cfg, device=device)
        queue_set = ModelUpdateQueueSet()

        model_trainer = SCDRTrainerProcess(queue_set, cdr_model, cfg.exp_params.dataset, cfg_path, cfg,
                                           result_save_dir, device=device, log_path=log_path)

        ex.start_parallel_scdr(queue_set, model_trainer)
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

    # ex = StreamingEx(cfg, custom_indices, result_save_dir)

    ex = StreamingExProcess(cfg, custom_indices, result_save_dir, )

    start(ex)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default=PAR_SCDR,
                        choices=[ATSNE, SIPCA, XTREAMING, SCDR, RTSCDR, PAR_SCDR])
    parser.add_argument("--indices_dir", type=str, default=r"H:\Projects\流数据\Data\indices\single_cluster")
    parser.add_argument("-Xmx", type=str, default="102400m")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg_path = ConfigInfo.MODEL_CONFIG_PATH.format(args.method)
    cfg = get_config()
    cfg.merge_from_file(cfg_path)
    result_save_dir = "results/{}/".format(args.method)

    # random_training()
    custom_indices_path = os.path.join(args.indices_dir, "{}.npy".format(cfg.exp_params.dataset))
    custom_indices_training(custom_indices_path)
    # stream_rate_ex()
    # cluster_composite_ex()
