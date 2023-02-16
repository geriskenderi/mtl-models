import os
import shutil
import argparse

from sklearn.model_selection import train_test_split

def main(args):
    dataset_path = args.data_path

    elems = len(os.listdir(dataset_path))
    train_count = round(0.80 * elems) 
    test_count = round(0.20 * elems)
    assert(train_count + test_count == elems)
    
    trainset, testset = train_test_split(os.listdir(dataset_path), 
                                         train_size=train_count, 
                                         test_size=test_count,
                                         random_state=21)

    train_path = os.path.join(dataset_path, "train")
    os.makedirs(train_path, exist_ok=True)
    for img in sorted(trainset):
        src_img_path = os.path.join(dataset_path, img)
        dst_img_path = os.path.join(dataset_path, "train", img)
        shutil.copyfile(src_img_path, dst_img_path)

    test_path = os.path.join(dataset_path, "test")
    os.makedirs(test_path, exist_ok=True)
    for img in sorted(testset):
        src_img_path = os.path.join(dataset_path, img)
        dst_img_path = os.path.join(dataset_path, "test", img)
        shutil.copyfile(src_img_path, dst_img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='')

    args = parser.parse_args()
    main(args)
