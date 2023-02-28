#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import argparse

from sklearn.model_selection import train_test_split


def main(args):
    dataset_path = args.data_path

    elems = len(os.listdir(dataset_path))
    train_count = round(args.train_size * elems)
    test_count = round(args.test_size * elems)
    assert(train_count + test_count == elems)

    trainset, testset = train_test_split(os.listdir(dataset_path),
                                         train_size=train_count,
                                         test_size=test_count,
                                         random_state=args.seed)

    train_path = os.path.join(dataset_path, "train")
    os.makedirs(train_path, exist_ok=True)
    for img in sorted(trainset):
        src_img_path = os.path.join(dataset_path, img)
        dst_img_path = os.path.join(dataset_path, "train", img)
        shutil.copyfile(src_img_path, dst_img_path)

    test_path = os.path.join(dataset_path, "val")
    os.makedirs(test_path, exist_ok=True)
    for img in sorted(testset):
        src_img_path = os.path.join(dataset_path, img)
        dst_img_path = os.path.join(dataset_path, "val", img)
        shutil.copyfile(src_img_path, dst_img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=float, default='21')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--train_size', type=float, default='0.80')
    parser.add_argument('--test_size', type=float, default='0.20')

    args = parser.parse_args()
    main(args)
