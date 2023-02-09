import argparse
from multiprocessing import Queue

import numpy as np
import os

from dataset.streaming_data_mock import SimulatedStreamingData
from model.scdr.model_trainer import SCDRTrainer, SCDRTrainerProcess
from experiments.streaming_experiment import StreamingEx, StreamingExProcess
from model.dr_models.ModelSets import MODELS
from utils.constant_pool import ConfigInfo, SIPCA, ATSNE, XTREAMING, SCDR, STREAM_METHOD_LIST, INE, SISOMAPPP, ILLE
from utils.common_utils import get_config
from utils.queue_set import ModelUpdateQueueSet, StreamDataQueueSet

device = "cuda:0"
log_path = "logs/logs.txt"


def start(ex, recv_args):
    if recv_args.method == ATSNE:
        # ==============1. at-SNE model=====================
        return ex.start_atSNE()
    elif recv_args.method == ILLE:
        return ex.start_ille()
    elif recv_args.method == SIPCA:
        # ==============2. siPCA model=====================
        if recv_args.parallel:
            return ex.start_parallel_spca()
        else:
            return ex.start_siPCA()
    elif recv_args.method == XTREAMING:
        # ==============3. Xtreaming model=====================
        if recv_args.parallel:
            return ex.start_parallel_xtreaming()
        else:
            return ex.start_xtreaming()
    elif recv_args.method == INE:
        # ==============4. INE model=====================
        if recv_args.parallel:
            return ex.start_parallel_ine()
        else:
            return ex.start_ine()
    elif recv_args.method == SISOMAPPP:
        if recv_args.parallel:
            return ex.start_parallel_sisomap()
        else:
            return ex.start_sisomap()
    elif recv_args.method == SCDR:
        assert isinstance(ex, StreamingExProcess)
        cdr_model = MODELS["LwF_CDR"](cfg, device=device)
        model_update_queue_set = ModelUpdateQueueSet()

        model_trainer = SCDRTrainerProcess(model_update_queue_set, cdr_model, cfg.exp_params.dataset,
                                           cfg_path, cfg, result_save_dir, device=device, log_path=log_path)
        # return ex.start_parallel_scdr(model_update_queue_set, model_trainer)
        return ex.start_full_parallel_scdr(model_update_queue_set, model_trainer)
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


def custom_indices_training(configs, custom_indices_path, recv_args, res_save_dir):
    # custom_indices_path = os.path.join(ConfigInfo.CUSTOM_INDICES_DIR, "{}.npy".format(args.dataset))
    custom_indices = np.load(custom_indices_path, allow_pickle=True)

    stream_data_queue_set = Queue()
    start_data_queue = Queue()
    data_generator = SimulatedStreamingData(configs.exp_params.dataset, configs.exp_params.stream_rate,
                                            stream_data_queue_set, start_data_queue, custom_indices)
    ex = StreamingExProcess(configs, custom_indices, res_save_dir, stream_data_queue_set, start_data_queue, data_generator)

    data_generator.start()

    return start(ex, recv_args)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default=XTREAMING,
                        choices=[SIPCA, XTREAMING, INE, SISOMAPPP, SCDR])
    parser.add_argument("--indices_dir", type=str, default=r"../../Data/new/indices_seq")
    parser.add_argument("--parallel", type=bool, default=False)
    parser.add_argument("-Xmx", type=str, default="102400m")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg_path = ConfigInfo.MODEL_CONFIG_PATH.format(args.method)
    cfg = get_config()
    cfg.merge_from_file(cfg_path)
    result_save_dir = "results/{}/".format(args.method)

    custom_indices_path = os.path.join(args.indices_dir, "{}_FD.npy".format(cfg.exp_params.dataset))
    custom_indices_training(cfg, custom_indices_path, args, result_save_dir)

    # suffix_list = ["TI", "FV", "TV"]
    # suffix_list = ["TI", "TV"]
    # for item in suffix_list:
    #     custom_indices_path = os.path.join(args.indices_dir, "{}_{}.npy".format(cfg.exp_params.dataset, item))
    #     for i in range(1):
    #         # cfg.exp_params.make_animation = i == 0
    #         custom_indices_training(custom_indices_path)
