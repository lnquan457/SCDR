#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.datasets import MyTextDataset, MyImageDataset


def simple_transforms():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform


def get_dataset(dataset_name, root_dir, train, transform=None, is_image=True):
    if is_image:
        dataset = MyImageDataset(dataset_name, root_dir, train, transform)
    else:
        dataset = MyTextDataset(dataset_name, root_dir, train)
    return dataset


def get_dataloaders(dataset_name, root_dir, batch_size, shuffle, transform=None, is_image=True):
    train_dataset = get_dataset(dataset_name, root_dir, True, transform, is_image)
    # test_dataset = get_dataset(dataset_name, root_dir, False, transform, is_image)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, None


