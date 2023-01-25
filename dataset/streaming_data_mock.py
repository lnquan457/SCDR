import time

import numpy as np

from dataset.datasets import load_local_h5_by_path
import os

from utils.constant_pool import ConfigInfo, ProjectSettings
from utils.common_utils import date_time2timestamp
from multiprocessing import Process, Queue


def resort_label(label_seq):
    # 按照cls出现的顺序重新分配标签
    unique_cls, show_seq = np.unique(label_seq, return_index=True)
    color_list = []
    for i, item in enumerate(unique_cls):
        indices = np.argwhere(label_seq == item).squeeze()
        label_seq[indices] = i
        color_list.extend([ProjectSettings.LABEL_COLORS[i]] * len(indices))
    return label_seq.astype(int), color_list


class RealStreamingData(Process):
    def __init__(self, dataset_name, queue_set):
        self.name = "真实流数据产生进程"
        Process.__init__(self, name=self.name)
        self.queue_set = queue_set
        self.data_file_path = os.path.join(ConfigInfo.DATASET_CACHE_DIR, dataset_name + ".h5")
        # 格式为：time_info,data_nums，time_info 统一为%Y-%m-%d %H:%M:%S的格式
        self.time_file_path = os.path.join(ConfigInfo.TIME_INFO_CACHE_DIR, dataset_name + ".txt")
        self.data_index = 0
        self.stop_flag = False
        # 二元组，表示每个时间戳产生的数据个数
        self.time_data_num = []
        self._load_data()

    def _load_data(self):
        self.data, self.targets = load_local_h5_by_path(self.data_file_path, ['x', 'y'])
        time_file = open(self.time_file_path)
        if self.targets is None:
            self.seq_label = None
            self.seq_color = None
        else:
            self.seq_label, self.seq_color = resort_label(self.targets)
        for line in time_file.readlines():
            date_time, data_num = line.split(",")
            timestamp = date_time2timestamp(date_time)
            self.time_data_num.append([timestamp, int(data_num)])

    def run(self) -> None:
        while not self.stop_flag:
            self.queue_set.start_flag_queue.get(block=True)
            if self.data_index > 0:
                self.data_index = 0

            for i, (cur_timestamp, cur_data_num) in enumerate(self.time_data_num):
                cur_data = []
                for j in range(self.data_index, self.data_index + cur_data_num):
                    cur_data.append([self.data[j], None if self.targets is None else self.targets[j]])
                self.queue_set.training_data_queue.put(cur_data)
                self.data_index += cur_data_num

                if i < len(self.time_data_num) - 1:
                    time.sleep(self.time_data_num[i + 1][0] - cur_timestamp)

            self.queue_set.stop_flag_queue.put(True)


class SimulatedStreamingData(Process):
    def __init__(self, dataset_name, stream_rate, queue_set, custom_seq=None):
        self.name = "模仿流数据产生进程"
        Process.__init__(self, name=self.name)
        self.queue_set = queue_set
        self.dataset_name = dataset_name
        self.stream_rate = stream_rate
        self.stop_flag = False
        self.data_index = 0

        self.data_file_path = os.path.join(ConfigInfo.DATASET_CACHE_DIR, dataset_name + ".h5")
        self.data, self.targets = load_local_h5_by_path(self.data_file_path, ['x', 'y'])
        self.n_samples = self.data.shape[0]
        initial_num = 0
        if custom_seq is None:
            self.custom_seq = np.arange(0, self.n_samples, 1)
            self.time_step_num = int(np.ceil(len(self.custom_seq) / self.stream_rate))
        else:
            if not isinstance(custom_seq[0], int):
                initial_indices, stream_indices = custom_seq
                self.custom_seq = np.concatenate([initial_indices, stream_indices])
                initial_num = len(initial_indices)
                self.time_step_num = int(np.ceil(len(stream_indices) / self.stream_rate)) + 1
            else:
                self.custom_seq = custom_seq
                self.time_step_num = int(np.ceil(len(self.custom_seq) / self.stream_rate))

        if self.targets is None:
            self.seq_label = None
            self.seq_color = None
        else:
            self.seq_label, self.seq_color = resort_label(self.targets[self.custom_seq])

        stream_num = self.n_samples - initial_num

        self.data_num_list = np.ones(shape=(stream_num // stream_rate)) * stream_rate
        left = stream_num - np.sum(self.data_num_list)
        if left > 0:
            self.data_num_list = np.append(self.data_num_list, left)

        if initial_num > 0:
            self.data_num_list = np.append(initial_num, self.data_num_list)

        self.data_num_list = self.data_num_list.astype(int)

    def run(self) -> None:
        idx = 0
        self.queue_set.start_flag_queue.get(block=True)
        print("start adding data!")
        if self.data_index > 0:
            self.data_index = 0
        while not self.stop_flag:

            while not self.queue_set.data_queue.empty():
                pass

            # if self.data_index > 800:
            #     time.sleep(0.05)

            if idx >= len(self.data_num_list):
                self.queue_set.stop_flag_queue.put(True)
                break

            cur_data_num = self.data_num_list[idx]

            # TODO：调试用
            # if idx > 500:
            #     break
            # ================
            cur_data = []
            for j in self.custom_seq[self.data_index:self.data_index + cur_data_num]:
                cur_data.append([self.data[j], None if self.targets is None else self.targets[j]])
            self.queue_set.data_queue.put(cur_data)

            self.data_index += cur_data_num
            idx += 1


class StreamingDataMock:
    def __init__(self, dataset_name, num_per_step, seq_indices):
        self.seq_indices = seq_indices
        self.data_file_path = os.path.join(ConfigInfo.DATASET_CACHE_DIR, dataset_name + ".h5")
        self.data, self.targets = load_local_h5_by_path(self.data_file_path, ['x', 'y'])
        self._n_samples = self.data.shape[0]

        if seq_indices is None:
            self.custom_seq = np.arange(0, self._n_samples, 1)
        else:
            if not isinstance(seq_indices[0], int):
                initial_indices, stream_indices = seq_indices
                self.custom_seq = np.concatenate([initial_indices, stream_indices])
            else:
                self.custom_seq = seq_indices

        if self.targets is None:
            self.seq_label = None
            self.seq_color = None
        else:
            self.seq_label, self.seq_color = resort_label(self.targets[self.custom_seq])

        self.num_per_step = num_per_step
        self.time_step = 0
        self.data_index = 0
        self.time_step_num = int(np.ceil(len(seq_indices) / self.num_per_step)) if seq_indices is not None else 0
        self.cls2idx = {}

    def next_time_data(self):
        from_idx, to_idx = self.data_index, min(self.data_index + self.num_per_step, self.data.shape[0])
        cur_data = self.data[self.seq_indices[from_idx:to_idx]]
        cur_label = self.targets[self.seq_indices[from_idx:to_idx]]

        self.time_step += 1
        self.data_index += to_idx - from_idx
        end = self.time_step >= self.data.shape[0]
        return cur_data, cur_label, end


class StreamingDataMock2Stage(StreamingDataMock):
    def __init__(self, dataset_name, initial_cls, initial_ratio, num_per_step, seq_indices=None,
                 drop_last=True):
        StreamingDataMock.__init__(self, dataset_name, num_per_step, seq_indices)

        self.initial_cls = initial_cls
        self.initial_ratio = initial_ratio
        self.drop_last = drop_last
        self.initial_indices = None
        self.after_indices = None
        self.time_data_seq = None
        self.history_data = None
        self.history_label = None

        if seq_indices is None:
            self._generate_indices_seq()
        else:
            self.initial_indices, self.after_indices = seq_indices

        self._generate_time_data_seq()

    def _generate_indices_seq(self):
        initial_cls_indices = []
        all_indices = np.arange(0, self.data.shape[0], 1)
        for cls in self.initial_cls:
            cur_indices = np.argwhere(self.targets == cls).squeeze()
            initial_cls_indices.extend(cur_indices)

        initial_cls_indices = np.array(initial_cls_indices, dtype=int)
        np.random.shuffle(initial_cls_indices)
        self.initial_indices = initial_cls_indices[:int(self.initial_ratio * len(initial_cls_indices))]
        self.after_indices = np.setdiff1d(all_indices, self.initial_indices).astype(int)
        np.random.shuffle(self.after_indices)

    def _generate_time_data_seq(self):
        self.time_step_num = int(np.ceil(len(self.after_indices) / self.num_per_step))
        self.time_data_seq = []
        for i in range(self.time_step_num):
            if i == self.time_step_num - 1 and self.drop_last:
                self.time_step_num -= 1
                break
            self.time_data_seq.append(
                self.after_indices[i * self.num_per_step:min((i + 1) * self.num_per_step, len(self.after_indices))])
        self.time_data_seq = np.array(self.time_data_seq)
        self.time_step_num += 1

    def _get_initial_data(self):
        self.history_data = self.data[self.initial_indices]
        self.history_label = self.targets[self.initial_indices]
        return self.history_data, self.history_label, False

    def next_time_data(self):
        if self.time_step == 0:
            self.time_step += 1
            return self._get_initial_data()

        cur_data, cur_label = self.data[self.time_data_seq[self.time_step-1].astype(int)], \
                              self.targets[self.time_data_seq[self.time_step-1].astype(int)]
        self.history_data = np.concatenate([self.history_data, cur_data], axis=0)
        self.history_label = np.concatenate([self.history_label, cur_label], axis=0)
        self.time_step += 1
        end = self.time_step >= self.data.shape[0]
        return cur_data, cur_label, end
