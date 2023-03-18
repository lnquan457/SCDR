import h5py
from matplotlib import animation
from matplotlib.animation import FuncAnimation

from utils.constant_pool import FINAL_DATASET_LIST
import matplotlib.pyplot as plt
import numpy as np
import os


def _make_embedding_video(save_path, embeddings_list, labels_list, x_min, x_max, y_min, y_max):
    def _loose(d_min, d_max, rate=0.05):
        scale = d_max - d_min
        d_max += np.abs(scale * rate)
        d_min -= np.abs(scale * rate)
        return d_min, d_max

    # l_x_min, l_x_max = _loose(self._x_min, self._x_max)
    # l_y_min, l_y_max = _loose(self._y_min, self._y_max)

    fig, ax = plt.subplots()

    def update(idx):
        if idx % 100 == 0:
            print("frame", idx)

        cur_step_embeddings = embeddings_list[idx]
        cur_step_labels = labels_list[idx]

        ax.cla()
        # ax.set(xlim=(l_x_min, l_x_max), ylim=(l_y_min, l_y_max))
        # ax.set_aspect('equal')
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        # fig.tight_layout()

        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
        ax.scatter(x=cur_step_embeddings[:, 0], y=cur_step_embeddings[:, 1],
                   c=cur_step_labels, s=2, cmap="tab10")
        ax.set_title("Process: %2.2f" % (100 * idx / FRAMES) + "%")

    ani = FuncAnimation(fig, update, frames=FRAMES, interval=1/FPS, blit=False)
    ani.save(save_path, writer='ffmpeg', dpi=300)


if __name__ == '__main__':
    raw_data_dir = r"D:\Projects\流数据\Data\H5 Data"
    indices_dir = r"D:\Projects\流数据\Data\new\indices_seq"
    embedding_data_dir = r"D:\Projects\流数据\Evaluation\GIF\原始数据"
    # method_list = ["sPCA", "Xtreaming", "SIsomap++", "INE"]
    method_list = ["INE"]
    # dataset_list = FINAL_DATASET_LIST
    dataset_list = ["HAR_2"]
    FPS = 35
    FRAMES = 200
    situation = "PD"
    sta_t = 1
    eval_step = 1
    window_size = 5000

    print(animation.writers.list())

    for method in method_list:
        method_dir = os.path.join(embedding_data_dir, method)

        for dataset in dataset_list:
            dataset_dir = os.path.join(method_dir, dataset)
            cur_embeddings_dir = os.path.join(dataset_dir, "eval_embeddings")

            with h5py.File(os.path.join(raw_data_dir, "{}.h5".format(dataset)), "r") as hf:
                x = np.array(hf['x'], dtype=float)
                y = np.array(hf['y'], dtype=int)

            pre_timestep = 1
            cur_time_step = 1
            initial_indices, after_indices = np.load(os.path.join(indices_dir, "{}_{}.npy".format(dataset, situation)),
                                                     allow_pickle=True)
            labels = y[np.concatenate([initial_indices, after_indices])]
            initial_num = len(initial_indices)
            if initial_num > window_size:
                sta_idx = initial_num - window_size
            else:
                sta_idx = 0
            end_idx = initial_num
            pre_embeddings = None
            pre_labels = None

            total_embedding_list = []
            total_labels_list = []

            while True:
                cur_time_step += eval_step
                time_step_e_file = "t{}.npy".format(cur_time_step)
                t_step = cur_time_step - 1
                if not os.path.exists(os.path.join(cur_embeddings_dir, time_step_e_file)) or t_step > FRAMES:
                    break

                cur_embeddings = np.load(os.path.join(cur_embeddings_dir, time_step_e_file), allow_pickle=True)[1]

                new_data_num = 1
                end_idx += new_data_num
                diff = end_idx - sta_idx - window_size
                if diff > 0:
                    if method == "Xtreaming":
                        diff = max(0, diff - 98)

                    sta_idx += diff
                    if method in ["sPCA", "Xtreaming"]:
                        sta_idx = max(0, sta_idx - 1)
                cur_labels = labels[sta_idx:end_idx]

                if method in ["sPCA", "Xtreaming"] and len(cur_labels) > cur_embeddings.shape[0]:
                    cur_labels = cur_labels[:cur_embeddings.shape[0]]
                if method == "SCDR" and len(cur_labels) < cur_embeddings.shape[0]:
                    cur_embeddings = cur_embeddings[-len(cur_labels):]

                show_labels = cur_labels
                if method == "Xtreaming" and pre_labels is not None and t_step % 200 == 0:
                    show_labels = pre_labels

                total_embedding_list.append(cur_embeddings)
                total_labels_list.append(show_labels)
                print("Read t{} embedding".format(t_step))

                pre_timestep = t_step
                pre_embeddings = cur_embeddings
                pre_labels = cur_labels

            _make_embedding_video(os.path.join(dataset_dir, "{}.mp4".format(dataset)), total_embedding_list,
                                  total_labels_list, 0, 0, 0, 0)
