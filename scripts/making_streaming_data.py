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


def partial_drift(cls, cls_counts, labels, init_manifold_rate=0.5, init_data_rate=0.5):
    init_manifold_num = int(len(cls) * init_manifold_rate)
    init_data_indices, stream_data_indices = no_drift(cls[:init_manifold_num], cls_counts[:init_manifold_num], labels,
                                                      init_data_rate=init_data_rate)[:2]

    for i in range(init_manifold_num, len(cls)):
        cur_indices = np.where(labels == cls[i])[0]
        stream_data_indices.extend(cur_indices)

    np.random.shuffle(stream_data_indices)
    return init_data_indices, stream_data_indices, init_manifold_num, len(cls) - init_manifold_num

def full_drift(cls, cls_counts, labels, init_manifold_rate=0.3):
    init_data_indices = []
    stream_data_indices = []
    init_manifold_num = int(len(cls) * init_manifold_rate)
    ttt_indices = np.arange(len(cls))
    np.random.shuffle(ttt_indices)
    cls = cls[ttt_indices]
    cls_counts = cls_counts[ttt_indices]

    for i in range(init_manifold_num):
        cur_indices = np.where(labels == cls[i])[0]
        init_data_indices.extend(cur_indices)

    for i in range(init_manifold_num, len(cls)):
        cur_indices = np.where(labels == cls[i])[0]
        stream_data_indices.extend(cur_indices)

    np.random.shuffle(init_data_indices)
    np.random.shuffle(stream_data_indices)
    return init_data_indices, stream_data_indices, init_manifold_num, len(cls) - init_manifold_num


func_dict = {
    "ND": no_drift,
    "PD": partial_drift,
    "FD": full_drift
}


if __name__ == '__main__':
    dataset_dir = r"D:\Projects\流数据\Data\new"
    save_dir = os.path.join(dataset_dir, "indices_seq")
    check_path_exists(save_dir)
    situation_list = ["ND", "PD", "FD"]
    for item in os.listdir(dataset_dir):
        if not str(item).endswith(".h5") or item != "shuttle.h5":
            continue
        data_name = item.split(".")[0]

        with h5py.File(os.path.join(dataset_dir, item), "r") as hf:
            x = np.array(hf['x'])
            y = np.array(hf['y'], dtype=int)
            unique_cls, cls_nums = np.unique(y, return_counts=True)

            for situation in situation_list:
                init_indices, stream_indices, init_cls_num, stream_new_cls_num = func_dict[situation](unique_cls, cls_nums, y)
                save_path = os.path.join(save_dir, "{}_{}.npy".format(data_name, situation))
                np.save(save_path, [init_indices, stream_indices])
                print("{}_{} -> Init Num: {} Stream Num: {} Init Cls: {} Stream New Cls: {}".format(
                    data_name, situation, len(init_indices), len(stream_indices), init_cls_num, stream_new_cls_num))


