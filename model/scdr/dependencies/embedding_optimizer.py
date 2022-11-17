import random
import time

import numpy as np
import scipy

from scipy import optimize
from utils.loss_grads import nce_loss_single
from utils.umap_utils import find_ab_params


def initial_embedding_with_weighted_mean(neighbor_sims, neighbor_embeddings):
    normed_neighbor_sims = neighbor_sims / np.sum(neighbor_sims)
    cur_initial_embedding = np.sum(normed_neighbor_sims[:, np.newaxis] * neighbor_embeddings, axis=0)[np.newaxis, :]
    return cur_initial_embedding


class EmbeddingOptimizer:
    def __init__(self, local_move_thresh, bfgs_update_thresh, neg_num=50, min_dist=0.1, temperature=0.15,
                 skip_opt=False, timeout_thresh=1.0):
        self.nce_opt_update_thresh = 10
        self.__neg_num = neg_num
        self.__local_move_thresh = local_move_thresh
        self.__bfgs_update_thresh = bfgs_update_thresh
        self.__temperature = temperature
        self.__a, self.__b = find_ab_params(1.0, min_dist)
        self.skip_opt = skip_opt
        self.skip_optimizer = SkipOptimizer(bfgs_update_thresh, timeout_thresh) if skip_opt else None

    def update_local_move_thresh(self, new_thresh):
        self.__local_move_thresh = new_thresh

    def update_bfgs_update_thresh(self, new_thresh):
        self.__bfgs_update_thresh = new_thresh
        if self.skip_opt:
            self.skip_optimizer.update_bfgs_thresh(new_thresh)

    def optimize_new_data_embedding(self, neighbor_sims, neighbor_embeddings, other_embeddings):
        initial_embedding = initial_embedding_with_weighted_mean(neighbor_sims, neighbor_embeddings)
        neg_embeddings = other_embeddings[random.sample(list(np.arange(other_embeddings.shape[0])), self.__neg_num)]
        res = scipy.optimize.minimize(nce_loss_single, initial_embedding, method="BFGS", jac=None,
                                      args=(
                                          neighbor_embeddings, neg_embeddings, self.__a, self.__b, self.__temperature),
                                      options={'gtol': 1e-5, 'disp': False, 'return_all': False, 'eps': 1e-10})
        return res.x

    def update_old_data_embedding(self, new_data_embedding, old_embeddings, update_indices, knn_indices,
                                  knn_dists, corr_target_sims, anchor_positions, replaced_indices,
                                  replaced_raw_weights):

        if self.skip_opt:
            old_embeddings = self._pre_skip_opt(old_embeddings, knn_indices)

        local_move_mask = np.linalg.norm(old_embeddings[update_indices] - new_data_embedding, axis=-1) \
                          < self.__local_move_thresh

        if self.skip_opt:
            update_indices, anchor_positions, replaced_indices, replaced_raw_weights, local_move_mask = \
                self._mid_skip_opt(update_indices, knn_dists[update_indices], anchor_positions, replaced_indices,
                                   replaced_raw_weights, local_move_mask)

        old_embeddings[update_indices] = self._update(update_indices, knn_indices[update_indices], new_data_embedding,
                                                      old_embeddings, corr_target_sims, anchor_positions,
                                                      replaced_indices, replaced_raw_weights, local_move_mask)

        if self.skip_opt:
            self._end_skip_opt()

        return old_embeddings

    def _pre_skip_opt(self, embeddings, knn_indices):
        assert self.skip_optimizer is not None
        timeouts_indices = self.skip_optimizer.get_timeouts_indices()
        neg_indices = random.sample(list(np.arange(embeddings.shape[0])), self.__neg_num)
        for i, item in enumerate(timeouts_indices):
            embeddings[item] = self._nce_optimize_step(embeddings[item], embeddings[knn_indices[item]],
                                                       embeddings[neg_indices])

        return embeddings

    def _mid_skip_opt(self, update_indices, knn_dists, anchor_positions, replaced_indices, replaced_raw_weights,
                      local_move_mask):

        assert self.skip_optimizer is not None
        embedding_update_mask = self.skip_optimizer.skip(update_indices, knn_dists, local_move_mask)

        update_indices = update_indices[embedding_update_mask]
        anchor_positions = anchor_positions[embedding_update_mask].astype(int)
        replaced_indices = replaced_indices[embedding_update_mask].astype(int)
        replaced_raw_weights = replaced_raw_weights[embedding_update_mask]
        local_move_mask = local_move_mask[embedding_update_mask]

        return update_indices, anchor_positions.astype(int), replaced_indices.astype(
            int), replaced_raw_weights, local_move_mask

    def _end_skip_opt(self):
        self.skip_optimizer.update_records()

    def update_all_skipped_data(self, embeddings, knn_indices):
        if not self.skip_opt:
            return
        update_indices = self.skip_optimizer.update_all_skipped_data()
        neg_indices = random.sample(list(np.arange(embeddings.shape[0])), self.__neg_num)

        for i, item in enumerate(update_indices):
            embeddings[item] = self._nce_optimize_step(embeddings[item], embeddings[knn_indices[item]],
                                                       embeddings[neg_indices])
        return embeddings

    def _update(self, update_indices, corr_knn_indices, new_embeddings, old_embeddings, target_sims, anchor_position,
                replaced_neighbor_indices, replaced_sims, local_move_mask):
        # 使用BFGS算法进行优化
        total_embeddings = np.concatenate([old_embeddings, new_embeddings], axis=0)
        total_n_samples = total_embeddings.shape[0]
        neg_indices = random.sample(list(np.arange(total_n_samples)), self.__neg_num)

        for i, item in enumerate(update_indices):
            if local_move_mask[i]:
                anchor_sims = target_sims[i, anchor_position[i]] / np.sum(target_sims[i])
                back = replaced_sims[i] * (total_embeddings[replaced_neighbor_indices[i]] - total_embeddings[item])
                total_embeddings[item] -= back

                move = anchor_sims * (new_embeddings - total_embeddings[item])
                total_embeddings[item] = total_embeddings[item] + move
            else:
                total_embeddings[item] = self._nce_optimize_step(total_embeddings[item],
                                                                 total_embeddings[corr_knn_indices[i]],
                                                                 total_embeddings[neg_indices])

        return total_embeddings[update_indices]

    def _nce_optimize_step(self, optimize_embedding, positive_embeddings, neg_embeddings):
        res = scipy.optimize.minimize(nce_loss_single, optimize_embedding,
                                      method="BFGS", jac=None,
                                      args=(positive_embeddings, neg_embeddings, self.__a, self.__b, self.__temperature),
                                      options={'gtol': 1e-5, 'disp': False, 'return_all': False, 'eps': 1e-10})
        optimized_e = res.x
        update_step = optimized_e - optimize_embedding
        # TODO:不像参数化方法，这种非参方法对NCE损失的鲁棒性比较差
        update_step[update_step > self.nce_opt_update_thresh] = 0
        update_step[update_step < -self.nce_opt_update_thresh] = 0
        optimize_embedding += update_step
        return optimize_embedding


class SkipOptimizer:
    def __init__(self, bfgs_update_thresh, timeout_thresh=1.0):
        self.timeout_thresh = timeout_thresh
        self.__bfgs_update_thresh = bfgs_update_thresh
        self.delayed_meta = None

        self.timeout_meta_indices = []
        self.updated_data_indices = []
        self.skipped_data_indices = []

    def update_bfgs_thresh(self, new_thresh):
        self.__bfgs_update_thresh = new_thresh

    def get_timeouts_indices(self):
        if self.delayed_meta is not None:
            self.timeout_meta_indices = np.where(time.time() - self.delayed_meta[:, 1] >= self.timeout_thresh)[0]
            ret = self.delayed_meta[self.timeout_meta_indices][:, 0].astype(int)
        else:
            self.timeout_meta_indices = []
            ret = []
        return ret

    def skip(self, update_indices, knn_dists, local_move_mask):
        optimize_mask = np.mean(knn_dists, axis=1) < self.__bfgs_update_thresh
        embedding_update_mask = (local_move_mask + optimize_mask) > 0
        self.updated_data_indices = update_indices[embedding_update_mask]
        self.skipped_data_indices = update_indices[~embedding_update_mask]
        return embedding_update_mask

    def update_records(self):
        left_delayed_meta = None
        if self.delayed_meta is not None:
            left_delayed_meta_indices = np.setdiff1d(np.arange(self.delayed_meta.shape[0]),
                                                     self.timeout_meta_indices)
            left_delayed_meta = self.delayed_meta[left_delayed_meta_indices]
            updated_data_indices, position_1, _ = \
                np.intersect1d(left_delayed_meta[:, 0], self.updated_data_indices, return_indices=True)

            pre_left_meta_indices = np.setdiff1d(np.arange(len(left_delayed_meta_indices)), position_1)
            left_delayed_meta = left_delayed_meta[pre_left_meta_indices]

        if left_delayed_meta is not None:
            if len(left_delayed_meta.shape) < 2:
                left_delayed_meta = left_delayed_meta[np.newaxis, :]

            skipped_data_indices = np.setdiff1d(self.skipped_data_indices, left_delayed_meta[:, 0])
            skipped_data_meta = np.zeros(shape=(len(skipped_data_indices), 2))
            skipped_data_meta[:, 0] = skipped_data_indices
            skipped_data_meta[:, 1] = time.time()

            total_skipped_meta = np.concatenate([left_delayed_meta, skipped_data_meta], axis=0)
        else:
            total_skipped_meta = np.zeros(shape=(len(self.skipped_data_indices), 2))
            total_skipped_meta[:, 0] = self.skipped_data_indices
            total_skipped_meta[:, 1] = time.time()

        self.delayed_meta = total_skipped_meta

    def update_all_skipped_data(self):
        if len(self.delayed_meta.shape) < 2:
            return [int(self.delayed_meta[0])]
        else:
            return self.delayed_meta[:, 0].astype(int)
