import numpy as np

from dataset.datasets import load_local_h5_by_path
import os

from utils.constant_pool import ConfigInfo


class StreamingDataMock:
    def __init__(self, dataset_name, num_per_step, seq_indices):
        self.seq_indices = seq_indices
        self.data_file_path = os.path.join(ConfigInfo.DATASET_CACHE_DIR, dataset_name + ".h5")
        self.data, self.targets = load_local_h5_by_path(self.data_file_path, ['x', 'y'])
        self.num_per_step = num_per_step
        self.history_data = None
        self.history_label = None
        self.time_step = 0
        self.data_index = 0
        self.time_step_num = int(np.ceil(len(seq_indices) / self.num_per_step)) if seq_indices is not None else 0
        self.seq_label = None
        self.cls2idx = {}

    def next_time_data(self):
        from_idx, to_idx = self.data_index, min(self.data_index + self.num_per_step, self.data.shape[0])
        cur_data = self.data[self.seq_indices[from_idx:to_idx]]
        cur_label = self.targets[self.seq_indices[from_idx:to_idx]]

        self.history_data = cur_data if self.history_data is None else np.concatenate([self.history_data, cur_data], axis=0)
        self.history_label = cur_label if self.history_label is None else np.concatenate([self.history_label, cur_label], axis=0)
        self.get_seq_label(np.copy(cur_label))
        self.time_step += 1
        self.data_index += to_idx - from_idx
        return cur_data, cur_label

    def get_seq_label(self, cur_label):
        unique_cls = np.unique(cur_label)
        for item in unique_cls:
            indices = np.argwhere(cur_label == item).squeeze()
            if item not in self.cls2idx.keys():
                num = len(self.cls2idx.keys())
                self.cls2idx[item] = num
                cur_label[indices] = num
            else:
                cur_label[indices] = self.cls2idx[item]
        if self.seq_label is None:
            self.seq_label = cur_label
        else:
            self.seq_label = np.concatenate([self.seq_label, cur_label])


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
        return self.history_data, self.history_label

    def next_time_data(self):
        if self.time_step == 0:
            return self._get_initial_data()

        cur_data, cur_label = self.data[self.time_data_seq[self.time_step].astype(int)], \
                              self.targets[self.time_data_seq[self.time_step].astype(int)]
        self.history_data = np.concatenate([self.history_data, cur_data], axis=0)
        self.history_label = np.concatenate([self.history_label, cur_label], axis=0)
        self.time_step += 1
        return cur_data, cur_label
