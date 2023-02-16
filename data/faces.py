import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class FACES(Dataset):
    def __init__(self, dataset_path, tv_transforms, partition):
        super().__init__()
        self.dataset_path = dataset_path

        images = []
        # ids = []
        ages = []
        genders = []
        expressions = []
        # picture_sets = []

        for img in sorted(os.listdir(os.path.join(dataset_path, partition))):
            # Read person ID, age, and expression from filename.
            img_labes = img.split("_")
            # ids.append(img_labes[0])
            ages.append(img_labes[1])
            genders.append(img_labes[2])
            expressions.append(img_labes[3])
            # picture_sets.append(img_labes[4].split('.')[0])

            # Save the image.
            images.append(img)

        # Prepare dataset specific information.
        # id_lbls = {x:e for e, x in enumerate(sorted(set(ids)))}
        ages_lbls = {x:e for e, x in enumerate(sorted(set(ages)))}
        gender_lbls = {x:e for e, x in enumerate(sorted(set(genders)))}
        expression_lbls = {x:e for e, x in enumerate(sorted(set(expressions)))}
        # pset_lbls = {x:e for e, x in enumerate(sorted(set(picture_sets)))}

        # id_lbl_encoded = [id_lbls[x] for x in ids]
        age_lbl_encoded = [ages_lbls[x] for x in ages]
        gender_lbl_encoded = [gender_lbls[x] for x in genders]
        expression_lbl_encoded = [expression_lbls[x] for x in expressions]
        # pset_lbl_encoded = [pset_lbls[x] for x in picture_sets]
        # self.y = np.stack([id_lbl_encoded, gender_lbl_encoded, age_lbl_encoded, expression_lbl_encoded, pset_lbl_encoded], axis=1)
        self.y = np.stack([gender_lbl_encoded, age_lbl_encoded, expression_lbl_encoded], axis=1)
        self.imgs = images
        self.tv_transforms = tv_transforms

        # MTL information.
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        # self.task_lbl_sizes = [len(set(ids)), len(set(genders)), len(set(ages)),
        #                        len(set(expressions)), len(set(picture_sets))]
        self.task_lbl_sizes = [len(set(genders)), len(set(ages)), len(set(expressions))]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        imgs = Image.open(self.dataset_path / self.imgs[index]).convert('RGB')

        imgs = self.tv_transforms(imgs)
        return imgs, self.y[index]
