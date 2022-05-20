#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import random
import time
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data.sampler import T_co


class CustomSampler(Sampler):
    def __init__(self, train_indices):
        Sampler.__init__(self, None)
        self.indices = train_indices
        self.random = True

    def update_indices(self, new_indices, is_random):
        self.indices = new_indices
        self.random = is_random

    def __iter__(self):
        if self.random:
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class DebiasedSampler(Sampler):
    def __init__(self, data, knn_indices, batch_size):
        Sampler.__init__(self, data)
        self.batch_size = batch_size
        self.knn_indices = knn_indices
        self.n_samples = len(data)
        self.batch_num = self.n_samples // self.batch_size
        self.indices = np.arange(0, self.n_samples, 1).astype(np.int)
        self.ret_indices = []

    def __iter__(self):
        if len(self.ret_indices) > 0:
            return iter(self.ret_indices)
        candidate_indices = self.indices.copy()
        candidate_indices = set(candidate_indices)
        wait_indices = set()
        ret_indices = []
        last_num = -1
        sta = time.time()
        while len(candidate_indices) > 0 or len(wait_indices) > 0:
            if len(candidate_indices) == 0:
                if len(ret_indices) == last_num:
                    ret_indices.extend(wait_indices)
                    break
                last_num = len(ret_indices)

                if len(ret_indices) // self.batch_size < self.batch_num:
                    left = len(ret_indices) % self.batch_size
                    wait_indices = wait_indices.union(ret_indices[-left:])
                    ret_indices = ret_indices[:-left]

                candidate_indices = set(wait_indices)
                wait_indices.clear()

            cur = list(candidate_indices)[random.randint(0, len(candidate_indices) - 1)]
            candidate_indices.remove(cur)
            ret_indices.append(cur)
            candidate_indices = candidate_indices.difference(self.knn_indices[cur])
            wait_indices = wait_indices.union(self.knn_indices[cur]).difference(ret_indices)
        if random.random() > 0.99:
            print("debiased batch time cost: ", time.time() - sta)
        self.ret_indices = ret_indices
        return iter(self.ret_indices)

    def __len__(self):
        return self.n_samples


class SubsetDebiasedSampler(DebiasedSampler):
    def __init__(self, indices, data, knn_indices, batch_size):
        DebiasedSampler.__init__(self, data, knn_indices, batch_size)
        self.indices = indices
