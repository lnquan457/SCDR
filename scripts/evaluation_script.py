import argparse
import os

from stream_main import custom_indices_training
from utils.common_utils import get_config
from utils.constant_pool import ConfigInfo, SIPCA, ATSNE, XTREAMING, SCDR, STREAM_METHOD_LIST, INE, SISOMAPPP, ILLE


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default=INE,
                        choices=[ILLE, SIPCA, XTREAMING, INE, SISOMAPPP, SCDR])
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

    method_list = [SIPCA, XTREAMING, INE, SISOMAPPP, SCDR]
    situation_list = ["ND", "FD", "PD"]
    dataset_list = ["usps", "chess", "cifar10", "dry_bean", "fashion_mnist", "har", "mnist", "news_popularity",
                    "retina", "sat", "shuttle"]

    for i, dataset_name in enumerate(dataset_list):
        cfg.exp_params.dataset = dataset_name
        for j, jtem in enumerate(situation_list):
            custom_indices_path = os.path.join(args.indices_dir, "{}_{}.npy".format(cfg.exp_params.dataset, jtem))
            custom_indices_training(custom_indices_path, args)

