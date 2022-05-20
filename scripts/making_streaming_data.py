import math
import os
import h5py
import numpy as np


def check_path_exists(t_path):
    if not os.path.exists(t_path):
        os.makedirs(t_path)


def make_initial_and_after():
    save_dir = "../../../Data/indices"
    initial_cls_ratio = 0.6
    initial_ratio = 0.8
    for ds in ds_list:
        with h5py.File(os.path.join(data_dir, "{}.h5".format(ds)), "r") as hf:
            x = np.array(hf['x'], dtype=float)
            y = np.array(hf['y'], dtype=int)

            unique_cls = np.unique(y)
            np.random.shuffle(unique_cls)
            cls_num = int(initial_cls_ratio * len(unique_cls))
            initial_cls = unique_cls[:cls_num]
            initial_indices = []

            for item in initial_cls:
                cur_indices = np.argwhere(y == item).squeeze()
                np.random.shuffle(cur_indices)
                cur_num = int(math.ceil(initial_ratio * len(cur_indices)))
                initial_indices.extend(cur_indices[:cur_num])

            after_indices = np.setdiff1d(np.arange(0, len(y), 1), initial_indices)

            save_path = os.path.join(save_dir, "{}_c{}_r{}.npy".format(ds, cls_num, initial_ratio))
            np.save(save_path, [initial_indices, after_indices])


def make_single_cluster_data(ds_name):
    save_dir = "../../../Data/indices/single_cluster"
    unique_cls, y = read_labels(ds_name)
    np.random.shuffle(unique_cls)
    indices = []
    for item in unique_cls:
        cur_indices = np.argwhere(y == item).squeeze()
        np.random.shuffle(cur_indices)
        indices.extend(cur_indices)

    check_path_exists(save_dir)
    save_path = os.path.join(save_dir, "{}.npy".format(ds_name))
    np.save(save_path, indices)


def make_recurring_single_cluster_data(ds_name):
    save_dir = "../../../Data/indices/single_cluster_recur"
    unique_cls, y = read_labels(ds_name)
    np.random.shuffle(unique_cls)
    recurring_time = 2
    cls_indices = {}

    for item in unique_cls:
        cur_indices = np.argwhere(y == item).squeeze()
        np.random.shuffle(cur_indices)
        cls_indices[item] = np.array(cur_indices, dtype=int)

    indices = []
    for i in range(recurring_time):
        for item in unique_cls:
            num = math.ceil(len(cls_indices[item]) / recurring_time)
            indices.extend(cls_indices[item][i*num:min((i+1)*num, len(cls_indices[item]))])

    check_path_exists(save_dir)
    save_path = os.path.join(save_dir, "{}_recur{}.npy".format(ds_name, recurring_time))
    np.save(save_path, indices)


def make_multi_cluster_data(ds_name, composite_nums):
    save_dir = "../../../Data/indices/multi_cluster"
    unique_cls, y = read_labels(ds_name)
    np.random.shuffle(unique_cls)
    if isinstance(composite_nums, int):
        composite_num_tmp = [composite_nums for i in range(len(unique_cls) // composite_nums)]
        composite_nums = composite_num_tmp

    indices = []
    idx = 0
    for i in range(len(composite_nums)):
        cur_indices = []
        for j in range(composite_nums[i]):
            cur_indices.extend(np.argwhere(y == unique_cls[idx]).squeeze())
            idx += 1
        cur_indices = np.array(cur_indices, dtype=int)
        np.random.shuffle(cur_indices)
        indices.extend(cur_indices)

    composite_nums_str = [str(item) for item in composite_nums]
    suffix = "_".join(composite_nums_str)
    check_path_exists(save_dir)
    save_path = os.path.join(save_dir, "{}_{}.npy".format(ds_name, suffix))
    np.save(save_path, indices)


def make_recurring_multi_cluster_data(ds_name, composite_nums, recurring_time):
    save_dir = "../../../Data/indices/multi_cluster_recur"
    unique_cls, y = read_labels(ds_name)
    np.random.shuffle(unique_cls)
    if isinstance(composite_nums, int):
        composite_num_tmp = [composite_nums for i in range(len(unique_cls) // composite_nums)]
        composite_nums = composite_num_tmp

    idx = 0
    group_indices = {}
    for i in range(len(composite_nums)):
        cur_indices = []
        for j in range(composite_nums[i]):
            cur_indices.extend(np.argwhere(y == unique_cls[idx]).squeeze())
            idx += 1

        cur_indices = np.array(cur_indices, dtype=int)
        np.random.shuffle(cur_indices)
        group_indices[i] = cur_indices

    indices = []
    for i in range(recurring_time):
        for item in group_indices:
            num = math.ceil(len(group_indices[item]) / recurring_time)
            indices.extend(group_indices[item][i*num:min((i+1)*num, len(group_indices[item]))])

    composite_nums_str = [str(item) for item in composite_nums]
    suffix = "_".join(composite_nums_str)
    check_path_exists(save_dir)
    save_path = os.path.join(save_dir, "{}_{}_recur{}.npy".format(ds_name, suffix, recurring_time))
    np.save(save_path, indices)


def read_labels(ds_name):
    hf = h5py.File(os.path.join(data_dir, "{}.h5".format(ds_name)), "r")
    y = np.array(hf['y'], dtype=int)
    unique_cls = np.unique(y)
    return unique_cls, y


if __name__ == '__main__':
    data_dir = "../../../Data/H5 Data"
    ds_list = ["isolet_subset", "stanford dogs_subset", "wifi", "wethers", "food", "texture", "usps_clear", "satimage",
               "animals_clear", "mnist_r_10000", "cifar10_ex_10000", "pendigits"]
    composite_num_list = [3, [3, 4], 2, 2, [3, 3, 3, 2], [3, 3, 3, 2], [3, 3, 4], 3, [3, 3, 4], [3, 3, 4], [3, 3, 4],
                          [3, 3, 4]]
    # make_initial_and_after()
    # for i, ds in enumerate(ds_list):
        # make_single_cluster_data(ds)
        # make_recurring_single_cluster_data(ds)
        # make_multi_cluster_data(ds, composite_num_list[i])
        # make_recurring_multi_cluster_data(ds, composite_num_list[i], 2)

    make_recurring_multi_cluster_data("isolet_subset", [6], 5)