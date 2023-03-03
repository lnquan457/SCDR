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
    device = "cuda:0"
    log_path = "logs/logs_2.txt"
    # method_list = [SIPCA, XTREAMING, INE, SISOMAPPP]
    # method_list = [XTREAMING]
    method_list = [SCDR]
    # method_list = [INE, SCDR, SISOMAPPP]
    # method_list = [XTREAMING, INE, SISOMAPPP]
    # method_list = [SCDR]
    test_time = 1
    # situation_list = ["ND", "FD", "PD"]
    situation = "FD"
    # dataset_list = ["sat", "HAR_2", "usps",  "mnist_fla", "shuttle", "arem", "basketball"]
    # dataset_list = ["sat", "HAR_2", "usps", "mnist_fla", "shuttle", "arem", "basketball", "electric_devices",
    #                 "texture", "Anuran Calls_8c"]
    # dataset_list = ["mnist_fla", "HAR_2", "arem", "basketball", "shuttle"]
    dataset_list = ["mnist_fla", "HAR_2", "arem", "basketball", "shuttle"]
    # dim_list = [36, 561, 256, 784, 9, 6, 6, 96, 40, 22]
    dim_list = [784, 561, 6, 6, 9]

    start_time = time_stamp_to_date_time_adjoin(time.time())
    excel_save_dir = r"D:\Projects\流数据\Code\SCDR\results\excel_res\{}_{}".format(start_time, situation)
    excel_headers = copy(METRIC_NAMES)
    excel_headers.extend(STEADY_METRIC_NAMES)
    excel_headers.append("Single Process Time")
    excel_headers.append("Total Process Time")
    excel_headers.append("Data Delay Time")

    for i, method_name in enumerate(method_list):
        print("Current method:", method_name)
        cfg = get_config()
        cfg_path = ConfigInfo.MODEL_CONFIG_PATH.format(method_name)
        cfg.merge_from_file(cfg_path)
        cfg.exp_params.window_size = 5000
        cfg.exp_params.vis_iter = 1000
        cfg.exp_params.eval_iter = 1000

        args.method = method_name
        result_save_dir = "results/{}/ex_{}_{}".format(args.method, start_time, situation)
        method_save_dir = os.path.join(excel_save_dir, method_name)
        if not os.path.exists(method_save_dir):
            os.makedirs(method_save_dir)

        for j, dataset_name in enumerate(dataset_list):
            print("Processing Data:", dataset_name)

            if method_name == SIPCA and "mnist" in dataset_name:
                continue
            if method_name == SCDR:
                cfg.exp_params.input_dims = dim_list[j]
            excel_save_path = os.path.join(method_save_dir, "{}.xlsx".format(dataset_name))
            total_res = []
            cfg.exp_params.dataset = dataset_name
            custom_indices_path = os.path.join(args.indices_dir, "{}_{}.npy".format(cfg.exp_params.dataset, situation))
            for e in range(test_time):
                metrics_list, avg_single_process_time, total_process_time, avg_data_delay_time = \
                    custom_indices_training(cfg, custom_indices_path, args, result_save_dir, cfg_path, device, log_path)
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



