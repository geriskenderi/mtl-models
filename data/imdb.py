#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch

import numpy as np

from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class IMDB(Dataset):
    def __init__(self, dataset_path, tv_transforms, partition_idx_path):
        super().__init__()
        # Read metadata from matlab file
        metadata = loadmat(os.path.join(dataset_path, 'imdb.mat'))['imdb'][0][0]
        gender = metadata[3].flatten().astype(int)
        age = metadata[10].flatten().astype(int)
        face_score = metadata[6].flatten()
        second_face_score = metadata[7].flatten()
        img_paths = np.array([str(x[0]) for x in metadata[2].flatten()])

        # Remove samples that do not contain a face, with more than 1 face or ages outside the range 0-99
        correct_idx = np.where((face_score != float('-inf')) & (np.isnan(second_face_score)) & ((age>=0) & (age<100)) & (~np.isnan(gender)) & (gender >= 0))[0]
        img_paths = img_paths[correct_idx]
        age = age[correct_idx]
        gender = gender[correct_idx]

        # # Train - test split
        # cid = metadata[9].flatten()[correct_idx]
        # unique_ids = np.unique(cid)
        # train_ids, test_ids = train_test_split(unique_ids, train_size=0.8, random_state=21)
        # train_idx, test_idx = np.where(np.isin(cid, train_ids))[0], np.where(np.isin(cid, test_ids))[0]
        # np.save('data/new_train_idx_v2.npy', train_idx)
        # np.save('data/new_val_idx_v2.npy', test_idx)
        
        # dataset
        self.dataset_path = dataset_path
        self.tv_transforms = tv_transforms
        partition_idx = np.load(partition_idx_path)
        self.imgs = img_paths[partition_idx]
        self.age = age[partition_idx]
        self.gender = gender[partition_idx]

        # mtl information
        self.y = np.stack([self.age, self.gender]).T
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        self.task_lbl_sizes = [100, 2]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        idx_path = os.path.join(self.dataset_path, self.imgs[index])
        img = Image.open(idx_path).convert('RGB').resize((256,256))

        return self.tv_transforms(img), torch.tensor(self.y[index])
