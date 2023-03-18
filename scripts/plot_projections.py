# coding=utf-8
import os.path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
import seaborn as sns
from model.scdr.dependencies.experiment import position_vis
import matplotlib.pyplot as plt

from utils.constant_pool import FINAL_DATASET_LIST


if __name__ == '__main__':
    data_dir = r"D:\Projects\流数据\Evaluation\原始数据\PD\0222"
    raw_dataset_dir = r"D:\Projects\流数据\Data\H5 Data"
    indices_dir = r"D:\Projects\流数据\Data\new\indices_seq"
    save_dir = r"D:\Projects\流数据\Evaluation\过程数据\散点图_2\color"
    # dataset_list = FINAL_DATASET_LIST
    dataset_list = ["arem", "basketball", "HAR_2", "mnist_fla", "shuttle"]
    method_list = ["sPCA", "Xtreaming", "SIsomap++", "INE", "SCDR"]

    selected_indices = [
        [100, 5100, 10100, 15100, 20100, 25100, 30100, 35500],
        [100, 6100, 12100, 18100, 24100, 30100, 36100, 43300],
        [100, 1000, 2000, 3000, 4000, 5000, 6000],
        [100, 7500, 14900, 22300, 29700, 37100, 44700],
        [100, 4100, 8100, 12100, 16100, 20100, 24100, 26300]
    ]

    # method_list = ["INE", "sPCA", "Xtreaming", "SCDR", "SIsomap++"]
    # method_list = ["SIsomap++"]
    # method_list = ["INE"]
    situation = "PD"
    window_size = 5000
    eval_step = 100

    for method in method_list:
        print("==========================={}=================================".format(method))
        method_dir = os.path.join(data_dir, method)
        for i, dataset in enumerate(dataset_list):
            print("Processing", dataset)
            if dataset == "mnist_fla" and method == "sPCA" and situation == "FD":
                continue
            dataset_dir = os.path.join(method_dir, dataset)

            with h5py.File(os.path.join(raw_dataset_dir, "{}.h5".format(dataset)), "r") as hf:
                y = np.array(hf['y'], dtype=int)

            initial_indices, after_indices = np.load(os.path.join(indices_dir, "{}_{}.npy".format(dataset, situation)),
                                                     allow_pickle=True)
            labels = y[np.concatenate([initial_indices, after_indices])]
            initial_num = len(initial_indices)

            for item in os.listdir(dataset_dir):
                if item.endswith(".xlsx"):
                    continue

                if initial_num > window_size:
                    sta_idx = initial_num - window_size
                else:
                    sta_idx = 0
                end_idx = initial_num

                pre_timestep = 1
                cur_time_step = 1
                item_dir = os.path.join(dataset_dir, item)
                embedding_dir = os.path.join(item_dir, "eval_embeddings")
                img_save_dir = os.path.join(item_dir, "imgs")
                gray_img_save_dir = os.path.join(item_dir, "gray_imgs")
                if not os.path.exists(gray_img_save_dir):
                    os.makedirs(gray_img_save_dir)
                pre_embeddings = None
                pre_labels = None
                while True:
                    cur_time_step += eval_step
                    time_step_e_file = "t{}.npy".format(cur_time_step - 1)
                    if not os.path.exists(os.path.join(embedding_dir, time_step_e_file)):
                        break

                    t_step = cur_time_step - 1
                    # save_path = os.path.join(img_save_dir, "t_{}.jpg".format(t_step))
                    # gray_save_path = os.path.join(gray_img_save_dir, "t_{}.jpg".format(t_step))

                    gray_save_dir = os.path.join(save_dir, dataset, method)
                    if not os.path.exists(gray_save_dir):
                        os.makedirs(gray_save_dir)

                    save_path = os.path.join(gray_save_dir, "t_{}.jpg".format(t_step))
                    gray_save_path = os.path.join(gray_save_dir, "t_{}.jpg".format(t_step))

                    cur_embeddings = np.load(os.path.join(embedding_dir, time_step_e_file), allow_pickle=True)[1]

                    new_data_num = t_step - pre_timestep
                    end_idx += new_data_num
                    diff = end_idx - sta_idx - window_size
                    if diff > 0:
                        if method == "Xtreaming":
                            diff = max(0, diff - 98)

                        sta_idx += diff
                        if method in ["sPCA", "Xtreaming"]:
                            sta_idx = max(0, sta_idx - 1)
                    cur_labels = labels[sta_idx:end_idx]
                    # print(sta_idx, end_idx, "**", len(cur_labels))
                    if method in ["sPCA", "Xtreaming"] and len(cur_labels) > cur_embeddings.shape[0]:
                        cur_labels = cur_labels[:cur_embeddings.shape[0]]
                    if method == "SCDR" and len(cur_labels) < cur_embeddings.shape[0]:
                        cur_embeddings = cur_embeddings[-len(cur_labels):]
                    print("TimeStep: {}".format(t_step), len(cur_labels), cur_embeddings.shape[0])

                    show_labels = cur_labels
                    if method == "Xtreaming" and pre_labels is not None and t_step % 200 == 0:
                        show_labels = pre_labels

                    pre_timestep = t_step
                    pre_embeddings = cur_embeddings
                    pre_labels = cur_labels

                    if cur_time_step - 1 not in selected_indices[i]:
                        continue

                    x = cur_embeddings[:, 0]
                    y = cur_embeddings[:, 1]

                    if dataset == "shuttle" and method in ["SIsomap++", "Xtreaming", "sPCA", "SCDR"]:
                        Percentile = np.percentile(x, [0, 25, 50, 75, 100])
                        thresh = 1.5
                        IQR = Percentile[3] - Percentile[1]
                        UpLimit = Percentile[3] + IQR * thresh
                        DownLimit = Percentile[1] - IQR * thresh

                        indices = np.array([True] * cur_embeddings.shape[0])
                        indices[x > UpLimit] = False
                        indices[x < DownLimit] = False

                        Percentile = np.percentile(y, [0, 25, 50, 75, 100])
                        IQR = Percentile[3] - Percentile[1]
                        UpLimit = Percentile[3] + IQR * thresh
                        DownLimit = Percentile[1] - IQR * thresh

                        indices[y > UpLimit] = False
                        indices[y < DownLimit] = False
                        x = x[indices]
                        y = y[indices]
                        show_labels = show_labels[indices]

                    plt.figure(figsize=(6, 6))
                    sns.scatterplot(x=x, y=y, hue=show_labels, s=5,
                                    palette="tab10", legend=False, alpha=1.0, linewidth=0)
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis("equal")
                    plt.axis('off')
                    # plt.title("{} embeddings".format(method), fontsize=18)
                    plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0.1)

                    plt.figure(figsize=(6, 6))
                    sns.scatterplot(x=x, y=y, s=5, legend=False, alpha=1.0,
                                    color="steelblue", linewidth=0)   # color="steelblue" #2F5597
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis("equal")
                    plt.axis('off')
                    # plt.savefig(gray_save_path, dpi=400, bbox_inches='tight', pad_inches=0.1)

                # break

