#! /usr/bin/env python3
import os
import random
import shutil
from datetime import datetime, timedelta

from skycam_utils import get_batch_dir, make_skycam_paths_json, get_img_time
from labeling_utils import get_uid, add_skycam_img, add_unlabeled_data, get_dataframe, save_df
from preprocess_skycam import preprocess_skycam_imgs

def make_batch_dir(task, batch_id, root='batch_data'):
    batch_path = root + '/' + get_batch_dir(task, batch_id)
    os.makedirs(batch_path, exist_ok=True)
    return batch_path


def select_samples():
    ...

samples = [
    {
        'skycam_type': 'SC2',
        'year': 2023,
        'month': 7,
        'day': 29
    },
    # {
    #     'skycam_type': 'SC2',
    #     'year': 2023,
    #     'month': 6,
    #     'day': 1
    # },
    # {
    #     'skycam_type': 'SC2',
    #     'year': 2020,
    #     'month': 6,
    #     'day': 24
    # },
    # {
    #     'skycam_type': 'SC2',
    #     'year': 2021,
    #     'month': 6,
    #     'day': 24
    # },
    # {
    #     'skycam_type': 'SC2',
    #     'year': 2023,
    #     'month': 5,
    #     'day': 24
    # },
    # {
    #     'skycam_type': 'SC2',
    #     'year': 2023,
    #     'month': 8,
    #     'day': 17
    # }
]

def add_skycam_data_to_img_df(img_df, skycam_paths):
    """Add skycam data to img_df """
    for skycam_dir in skycam_paths:
        # Only save original images to databases
        original_img_dir = skycam_paths[skycam_dir]['img_subdirs']['original']
        for fname in os.listdir(original_img_dir):
            if fname[-4:] == '.jpg':
                # Collect image features
                img_uid = get_uid(fname)
                skycam_type = fname.split('_')[0]
                t = get_img_time(fname)
                timestamp = (t - datetime(1970, 1, 1)) / timedelta(seconds=1)
                # Add entries to img_df
                add_skycam_img(img_df, fname, skycam_type, timestamp)


def build_batch(task, batch_id, do_zip=False):
    batch_path = make_batch_dir(task, batch_id)
    for sample in samples:
        preprocess_skycam_imgs(sample['skycam_type'],
                               sample['year'],
                               sample['month'],
                               sample['day'],
                               root=batch_path,
                               verbose=True)
    skycam_paths = make_skycam_paths_json(batch_path)
    img_df = get_dataframe('img')
    add_skycam_data_to_img_df(img_df, skycam_paths)
    save_df(img_df, 'img', None, batch_id, task, False, batch_path)
    if do_zip:
        zip_batch(task, batch_id)


def zip_batch(task, batch_id, root='batch_data_zipfiles'):
    batch_path = batch_path = make_batch_dir(task, batch_id)
    batch_dir = get_batch_dir(task, batch_id)
    batch_zip_name = f'{root}/{batch_dir}'
    if not os.path.exists(batch_zip_name + '.tar.gz'):
        print(f"Zipping {batch_dir}")
        os.makedirs(root, exist_ok=True)
        shutil.make_archive(batch_zip_name, 'gztar', batch_path)


if __name__ == '__main__':
    build_batch(task='cloud-detection', batch_id=0, do_zip=True)
