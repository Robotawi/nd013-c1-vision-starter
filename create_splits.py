import argparse
import glob
import os
import shutil
import random

import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records.
    The files should be moved to new folders in the same directory.
    This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """

    # tfrecord names
    # print(data_dir+'/*.tfrecord')
    tfrecords = [tfrecord for tfrecord in glob.glob(data_dir + '/*.tfrecord')]
    # print(tfrecords)

    # create directory with names test, train, and val
    test_path = os.path.join(data_dir, "test")
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")

    os.makedirs(test_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    train_split = 0.7 * len(tfrecords)
    test_split = 0.2 * len(tfrecords)
    val_split = 0.1 * len(tfrecords)

    print("Data split ratios:", train_split, test_split, val_split)

    for record_num, record_path in enumerate(tfrecords):
        # print(record_num, record_path)
        if record_num <= train_split:
            shutil.move(record_path, train_path)
        elif record_num > train_split and \
                record_num <= train_split + test_split:
            shutil.move(record_path, test_path)
        else:
            shutil.move(record_path, val_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True, help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
