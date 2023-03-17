#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import scipy.io

import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class CARS(Dataset):
    def __init__(self, dataset_path, tv_transforms, partition):
        super().__init__()

        self.partition = partition
        self.dataset_path = dataset_path

        fnames, class_ids, labels = [], [], []

        if self.partition == 'train':
            cars_annos = scipy.io.loadmat(os.path.join(dataset_path,
                                                       'devkit',
                                                       'cars_train_annos.mat'))
        else:
            cars_annos = scipy.io.loadmat(os.path.join(dataset_path,
                                                       'devkit',
                                                       'cars_test_annos_withlabels.mat'))
        annotations = cars_annos['annotations']
        annotations = np.transpose(annotations)

        for annotation in annotations:
            fname = annotation[0][5][0]
            fnames.append(fname)
            class_id = annotation[0][4][0][0]
            class_ids.append(class_id)
            labels.append('%04d' % (class_id,))

        # Prepare dataset specific information.
        cars_lbls = {x:e for e, x in enumerate(sorted(set(labels)))}
        car_lbl_encoded = [cars_lbls[x] for x in labels]

        self.imgs = fnames
        self.tv_transforms = tv_transforms
        self.y = np.stack([car_lbl_encoded], axis=1)

        # MTL information.
        self.num_tasks = self.y.shape[1]
        self.task_lbl_sizes = [len(set(labels))]
        self.task_ids = [i for i in range(self.num_tasks)]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        subfolder = str()
        if self.partition == 'train':
            subfolder = 'cars_train'
        else:
            subfolder = 'cars_test'
        imgs = Image.open(self.dataset_path / subfolder / self.imgs[index]).convert('RGB')

        imgs = self.tv_transforms(imgs)
        return imgs, self.y[index]
