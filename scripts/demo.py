# coding=utf-8
import os.path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

from model.scdr.dependencies.experiment import position_vis

if __name__ == '__main__':
    dataset_list = ["basketball", "HAR_2", "food", "sat", "har", "usps", "mnist_fla", "chess", "dry_bean",
                    "news_popularity", "shuttle", "arem", "activity_rec", "electric_devices"]
    data_dir = r"D:\Projects\流数据\Data\H5 Data"
    save_dir = r"D:\Projects\流数据\Data\data overview"
    for item in dataset_list:
        print(item)
        with h5py.File(os.path.join(data_dir, "{}.h5".format(item))) as hf:
            x = np.array(hf['x'], dtype=float)
            y = np.array(hf['y'], dtype=int)

            embedder = PCA()
            embeddings = embedder.fit_transform(x)
            save_path = os.path.join(save_dir, "pca_{}.jpg".format(item))
            position_vis(y, save_path, embeddings, title="{}-{}*{}-C{}".format(item, x.shape[0], x.shape[1],
                                                                               len(np.unique(y))))
