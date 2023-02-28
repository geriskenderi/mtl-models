#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CUB(Dataset):
    def __init__(self, dataset_path, tv_transforms, partition):
        super().__init__()
        
        self.partition = partition
        self.dataset_path = dataset_path
        self.tv_transforms = tv_transforms
        
        images = pd.read_csv(os.path.join(self.dataset_path, 'images.txt'),
                             sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.partition == 'train':
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        # MTL information.
        self.y = np.stack([self.data['target']], axis=1)
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        birds_lbls = np.unique(image_class_labels['target'])
        self.task_lbl_sizes = [len(set(birds_lbls))]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.dataset_path, 'images/', sample.filepath)
        # Targets start at 1 by default, so shift to 0.
        target = sample.target - 1 
        
        # img = self.loader(path)
        img = Image.open(path).convert('RGB')
        if self.tv_transforms is not None:
            img = self.tv_transforms(img)

        return img, self.y[target]
