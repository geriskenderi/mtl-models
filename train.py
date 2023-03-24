#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from pathlib import Path
from datetime import datetime
from torchvision import transforms as T
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger

from data.cub import CUB
from data.imdb import IMDB
from data.cars import CARS
from data.jaffe import JAFFE
from data.medic import MEDIC
from data.faces import FACES
from models.vgg import MTLVGG
from models.vgg_nddr import NDDRVGG

def main(args):
    # Seed 
    seed_everything(args.seed)

    # Define dataset and dataloaders
    data_path = Path(args.data_path)
    transforms = T.Compose([
        T.Resize((256,256)), 
        T.ToTensor()
    ])

    print('Loading datasets and dataloaders...')
    if args.dataset_name == 'imdb':
        train_file = data_path / args.train_file_path
        test_file = data_path / args.test_file_path
        trainset = IMDB(data_path, transforms, partition_idx_path=train_file)
        testset = IMDB(data_path, transforms, partition_idx_path=test_file)
    if args.dataset_name == 'jaffe':
        trainset = JAFFE(data_path, transforms, data_folder_path="train")
        testset = JAFFE(data_path, transforms, data_folder_path="val")
    if args.dataset_name == 'medic':
        train_file = data_path / args.train_file_path
        test_file = data_path / args.test_file_path
        trainset = MEDIC(data_path, transforms, partition_idx_path=train_file)
        testset = MEDIC(data_path, transforms, partition_idx_path=test_file)
    if args.dataset_name == 'faces':
        trainset = FACES(data_path, transforms, partition="train")
        testset = FACES(data_path, transforms, partition="val")
    if args.dataset_name == 'cub':
        train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            T.ColorJitter(brightness=.4, hue=.2),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()
        ])
        test_transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()
        ])
        trainset = CUB(data_path, train_transforms, partition="train")
        testset = CUB(data_path, test_transforms, partition="val")
    if args.dataset_name == 'cars':
        trainset = CARS(data_path, transforms, partition="train")
        testset = CARS(data_path, transforms, partition="val")

    # # Debug only
    # trainset = torch.utils.data.Subset(trainset, list(range(100)))
    # testset = torch.utils.data.Subset(testset, list(range(100)))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print('Done!')

    # Build model
    num_tasks = len(args.task_ids)
    assert len(args.task_output_sizes) >= num_tasks, 'Please provide one or more task ids and their corresponding output sizes (in order)'
    
    model = None
    if args.use_nddr:
        model = NDDRVGG(
            hidden_dim=args.hidden_dim,
            output_sizes=trainset.task_lbl_sizes,
            dataset_name=args.dataset_name,
            learning_rate=args.learning_rate
        )
    else:
        model = MTLVGG(
            hidden_dim=args.hidden_dim,
            num_tasks=num_tasks,
            task_ids=args.task_ids,
            output_sizes=trainset.task_lbl_sizes,
            dataset_name=args.dataset_name,
            learning_rate=args.learning_rate
        )
    
    # Training and evaluation
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
    run_name = f'VGG-{args.dataset_name}-multitask{num_tasks > 1}-usesNDDR_{args.use_nddr}-tasks{args.task_ids}'
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename=f'{run_name}'+ '{epoch}-' + dt_string,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        verbose=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"last_{run_name}"
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=args.epochs//3, verbose=True, mode="min")
    wandb_logger = WandbLogger(project="detangle-mtl", name=run_name, entity='vips4-univr')
    
    trainer = Trainer(
        # devices=2,
        # strategy="ddp",
        devices=[args.gpu_num],
        accelerator='gpu',
        max_epochs=args.epochs,
        check_val_every_n_epoch=2,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/media/data/gskenderi/imdb_crop/')
    parser.add_argument('--ckpt_path', type=str, default='/media/data/lcapogrosso/mtl-models_ckpt/')
    parser.add_argument('--train_file_path', type=str, default='new_train_idx_v2.npy')
    parser.add_argument('--test_file_path', type=str, default='new_val_idx_v2.npy')
    parser.add_argument('--dataset_name', type=str, default='imdb')
    parser.add_argument('--use_nddr', action='store_true')
    parser.add_argument('--task_ids', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--task_output_sizes', type=int, nargs='+', default=[3, 5])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.0001)     
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=21) 
    parser.add_argument('--gpu_num', type=int, default=0) 

    args = parser.parse_args()
    main(args)
