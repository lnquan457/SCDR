import argparse
import os
import time
from copy import copy

import numpy as np
import pandas as pd
from stream_main import custom_indices_training
from utils.common_utils import get_config, time_stamp_to_date_time_adjoin
from utils.constant_pool import ConfigInfo, SIPCA, ATSNE, XTREAMING, SCDR, STREAM_METHOD_LIST, INE, SISOMAPPP, ILLE, \
    METRIC_NAMES, STEADY_METRIC_NAMES


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default=INE,
                        choices=[SIPCA, XTREAMING, INE, SISOMAPPP, SCDR])
    parser.add_argument("--indices_dir", type=str, default=r"../../Data/new/indices_seq")
    parser.add_argument("--parallel", type=bool, default=False)
    parser.add_argument("-Xmx", type=str, default="102400m")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # method_list = [SIPCA, XTREAMING, INE, SISOMAPPP, SCDR]
    method_list = [SIPCA]
    test_time = 2
    # situation_list = ["ND", "FD", "PD"]
    situation = "FD"
    # dataset_list = ["usps", "chess", "cifar10", "dry_bean", "fashion_mnist", "har", "mnist", "news_popularity",
    #                 "retina", "sat", "shuttle"]
    dataset_list = ["food"]
    start_time = time_stamp_to_date_time_adjoin(time.time())
    result_save_dir = "results/{}/ex_{}".format(args.method, start_time)
    excel_save_dir = r"D:\Projects\流数据\Code\SCDR\results\excel_res\{}".format(start_time)
    excel_headers = copy(METRIC_NAMES)
    excel_headers.extend(STEADY_METRIC_NAMES)
    excel_headers.append("Single Process Time")
    excel_headers.append("Total Process Time")
    excel_headers.append("Data Delay Time")

    for i, method_name in enumerate(method_list):
        cfg = get_config()
        cfg_path = ConfigInfo.MODEL_CONFIG_PATH.format(method_name)
        cfg.merge_from_file(cfg_path)

        args.method = method_name
        method_save_dir = os.path.join(excel_save_dir, method_name)
        if not os.path.exists(method_save_dir):
            os.makedirs(method_save_dir)

        for j, dataset_name in enumerate(dataset_list):
            excel_save_path = os.path.join(method_save_dir, "{}.xlsx".format(dataset_name))
            total_res = []
            cfg.exp_params.dataset = dataset_name
            custom_indices_path = os.path.join(args.indices_dir, "{}_{}.npy".format(cfg.exp_params.dataset, situation))
            for e in range(test_time):
                metrics_list, avg_single_process_time, total_process_time, avg_data_delay_time = \
                    custom_indices_training(cfg, custom_indices_path, args, result_save_dir)
                cur_res = metrics_list
                cur_res.extend([avg_single_process_time, total_process_time, avg_data_delay_time])
                total_res.append(cur_res)

            total_res = np.array(total_res)
            avg = np.mean(total_res, axis=0)
            total_res = np.concatenate([total_res, avg[np.newaxis, :]], axis=0)
            res_dict = {}
            for t, ttem in enumerate(excel_headers):
                res_dict[ttem] = total_res[:, t]

            df = pd.DataFrame(res_dict)
            df.to_excel(excel_save_path)



