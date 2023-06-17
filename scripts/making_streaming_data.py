import math
import os
import h5py
import numpy as np


def check_path_exists(t_path):
    if not os.path.exists(t_path):
        os.makedirs(t_path)


"""
    三种情况的数据：
    1）所有新数据均来自与初始数据相同的流形，且不同流形的数据乱序产生；
    2）所有新数据均来自与初始数据不同的流形，且不同新流形的数据乱序产生；
    3）部分新数据来自与初始数据相同的流形，部分来自新的流形，且不同流形的数据乱序产生；
"""


def no_drift(cls, cls_counts, labels, init_data_rate=0.3):
    init_data_indices = []
    stream_data_indices = []

    for i in range(len(cls)):
        cur_indices = np.where(labels == cls[i])[0]
        np.random.shuffle(cur_indices)
        cur_num = int(cls_counts[i] * init_data_rate)
        init_data_indices.extend(cur_indices[:cur_num])
        stream_data_indices.extend(cur_indices[cur_num:])

    np.random.shuffle(init_data_indices)
    np.random.shuffle(stream_data_indices)
    return init_data_indices, stream_data_indices, len(cls), 0


def partial_drift(cls, cls_counts, labels, init_manifold_rate=0.5, init_data_rate=0.4):
    # cls = [0, 7, 5, 3, 9, 8, 6, 1, 2, 4]

    cls_counts = []
    for jtem in cls:
        cls_counts.append(len(np.where(labels == jtem)[0]))

    init_manifold_num = int(len(cls) * init_manifold_rate)
    init_data_indices, init_left_indices = no_drift(cls[:init_manifold_num], cls_counts[:init_manifold_num], labels,
                                                    init_data_rate=init_data_rate)[:2]
    avg_num = len(init_left_indices) // (len(cls) - init_manifold_num)

    init_left_indices = np.array(init_left_indices, dtype=int)

    init_left_counts = []
    init_left_indices_per_cls = []
    for i in range(init_manifold_num):
        cur_indices = np.where(labels[init_left_indices] == cls[i])[0]
        init_left_counts.append(len(cur_indices))
        init_left_indices_per_cls.append(init_left_indices[cur_indices])

    avg_left_counts = np.array(init_left_counts) / (len(cls) - init_manifold_num)
    avg_left_counts = avg_left_counts.astype(int)

    stream_data_indices = []
    idx = 0

    for i in range(init_manifold_num, len(cls)):
        cur_indices = np.where(labels == cls[i])[0]
        cur_total = list(cur_indices)

        for j in range(init_manifold_num):
            cur_total.extend(init_left_indices_per_cls[j][idx*avg_left_counts[j]:(idx+1)*avg_left_counts[j]])

        np.random.shuffle(cur_total)
        stream_data_indices.extend(cur_total)
        idx += 1

    return init_data_indices, stream_data_indices, init_manifold_num, len(cls) - init_manifold_num


def full_drift(cls, cls_counts, labels, init_manifold_rate=0.3, shuffle_stream=False):
    init_data_indices = []
    stream_data_indices = []
    init_manifold_num = int(math.ceil(len(cls) * init_manifold_rate))
    ttt_indices = np.arange(len(cls))
    np.random.shuffle(ttt_indices)
    cls = cls[ttt_indices]
    # cls = [1, 2, 8, 4, 6, 9, 3, 5, 7, 0]

    for i in range(init_manifold_num):
        cur_indices = np.where(labels == cls[i])[0]
        init_data_indices.extend(cur_indices)

    for i in range(init_manifold_num, len(cls)):
        cur_indices = np.where(labels == cls[i])[0]
        stream_data_indices.extend(cur_indices)

    np.random.shuffle(init_data_indices)
    if shuffle_stream:
        np.random.shuffle(stream_data_indices)
    return init_data_indices, stream_data_indices, init_manifold_num, len(cls) - init_manifold_num


func_dict = {
    "ND": no_drift,
    "PD": partial_drift,
    "FD": full_drift
}

if __name__ == '__main__':
    dataset_dir = r"D:\Projects\流数据\Data\H5 Data"
    save_dir = r"D:\Projects\流数据\Data\new\indices_seq"
    check_path_exists(save_dir)
    # dataset_list = ["usps_clear", "mnist_fla", "HAR_2", "arem", "shuttle", "basketball"]
    # dataset_list = ["usps_clear", "mnist_fla", "HAR_2", "arem", "shuttle", "basketball"]
    # dataset_list = ["mnist_fla_10000", "mnist_fla_20000", "usps_clear", "mnist_fla", "HAR_2", "arem", "shuttle", "basketball"]
    dataset_list = ["covid_twi"]
    situation_list = ["ND", "PD", "FD"]
    # situation_list = ["PD"]
    for item in dataset_list:
        data_name = item.split(".")[0]

        with h5py.File(os.path.join(dataset_dir, "{}.h5".format(item)), "r") as hf:
            x = np.array(hf['x'])
            y = np.array(hf['y'], dtype=int)
            unique_cls, cls_nums = np.unique(y, return_counts=True)

            for situation in situation_list:
                init_indices, stream_indices, init_cls_num, stream_new_cls_num = func_dict[situation](unique_cls,
                                                                                                      cls_nums, y)
                save_path = os.path.join(save_dir, "{}_{}.npy".format(data_name, situation))
                np.save(save_path, [init_indices, stream_indices])
                print("{}_{} -> Init Num: {} Stream Num: {} Init Cls: {} Stream New Cls: {}".format(
                    data_name, situation, len(init_indices), len(stream_indices), init_cls_num, stream_new_cls_num))
