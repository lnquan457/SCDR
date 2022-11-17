import os

import numpy as np
import h5py as hp


if __name__ == '__main__':
    data_dir = "../../Data/H5 Data"
    save_dir = "../../Data/indices/ex1116"
    dataset_list = ["food", "usps"]
    stream_type_list = ["TI", "FV", "TV"]
    INITIAL_NUM = 1000
    INITIAL_CLS_RATE = 0.5
    INITIAL_CLS_RATE_TV = 0.3

    for i, item in enumerate(dataset_list):
        print("正在处理数据集：", item)
        with hp.File(os.path.join(data_dir, "{}.h5".format(item))) as hf:
            x = np.array(hf['x'], dtype=float)
            y = np.array(hf['y'], dtype=int)
            unique_cls, cls_counts = np.unique(y, return_counts=True)
            cls_indices = []
            for cls in unique_cls:
                cls_indices.append(np.where(y == cls)[0])

            for j, jtem in enumerate(stream_type_list):
                print(" 流数据场景：", jtem)
                save_path = os.path.join(save_dir, "{}_{}.npy".format(item, jtem))
                initial_indices = []
                if j == 0:
                    initial_num_per_cls = (INITIAL_NUM * cls_counts / np.sum(cls_counts)).astype(int)
                    for k in range(len(unique_cls)):
                        initial_indices.extend(cls_indices[k][:initial_num_per_cls[k]])
                    left_indices = np.setdiff1d(np.arange(len(y)), initial_indices)
                    np.random.shuffle(left_indices)
                elif j == 1:
                    initial_cls_num = int(len(unique_cls) * INITIAL_CLS_RATE)
                    left_cls_num = len(unique_cls) - initial_cls_num
                    initial_cls_counts = cls_counts[:initial_cls_num]
                    initial_num_per_cls = (INITIAL_NUM * initial_cls_counts / np.sum(initial_cls_counts)).astype(int)
                    left_initial_cls_indices = []
                    for k in range(initial_cls_num):
                        initial_indices.extend(cls_indices[k][:initial_num_per_cls[k]])
                        left_initial_cls_indices.extend(cls_indices[k][initial_num_per_cls[k]:])

                    left_initial_cls_indices = np.array(left_initial_cls_indices, dtype=int)
                    np.random.shuffle(left_initial_cls_indices)
                    left_indices = []
                    initial_per_per = int(np.ceil(len(left_initial_cls_indices) / left_cls_num))
                    for k in range(left_cls_num):
                        left_indices.extend(cls_indices[k+initial_cls_num])
                        left_indices.extend(left_initial_cls_indices[k*initial_per_per:min((k+1)*initial_per_per,
                                                                                           len(left_initial_cls_indices))])

                else:
                    initial_cls_num = int(len(unique_cls) * INITIAL_CLS_RATE_TV)
                    initial_cls_counts = cls_counts[:initial_cls_num]
                    left_cls_num = len(unique_cls) - initial_cls_num
                    for k in range(initial_cls_num):
                        initial_indices.extend(cls_indices[k])

                    left_indices = []
                    for k in range(left_cls_num):
                        left_indices.extend(cls_indices[k + initial_cls_num])
                print("initial num: {} stream num: {}".format(len(initial_indices), len(left_indices)))
                np.save(save_path, [initial_indices, left_indices])
