import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class JAFFE(Dataset):
    def __init__(self, dataset_path, tv_transforms, data_folder_path):
        super().__init__()
        
        images = []
        ids = []
        expressions = []
        for expr in sorted(os.listdir(dataset_path / data_folder_path)):
            for file in sorted(os.listdir(dataset_path / data_folder_path / expr)):
                    # Read person id and expression from filename
                    labelz = file.split("-")
                    ids.append(labelz[0])
                    expressions.append(labelz[1][:-1])

                    # Read images
                    image = Image.open(dataset_path / data_folder_path / expr / file).convert('RGB')
                    images.append(image)
        
        # Prepare dataset specific information
        id_lbls = {x:e for e, x in enumerate(sorted(set(ids)))}
        expr_lbls = {x:e for e, x in enumerate(sorted(set(expressions)))}
        id_lbl_encoded = [id_lbls[x] for x in ids]
        expr_lbl_encoded = [expr_lbls[x] for x in expressions]
        self.y = np.stack([id_lbl_encoded, expr_lbl_encoded], axis=1)
        self.imgs = images
        self.tv_transforms = tv_transforms

        # mtl information
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        self.task_lbl_sizes = [10, 7]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        imgs = self.tv_transforms(self.imgs[index])
        return imgs, self.y[index]
