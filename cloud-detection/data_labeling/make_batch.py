#! /usr/bin/env python3
import os
import random
import shutil
from datetime import datetime, timedelta

from skycam_utils import get_batch_dir, make_skycam_paths_json, get_img_time, get_img_path
from labeling_utils import get_uid, add_skycam_img, get_dataframe, save_df
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
        'month': 8,
        'day': 1
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

def add_skycam_data_to_img_df(img_df, batch_id, skycam_paths, verbose):
    """Add skycam data to img_df """
    for skycam_dir in skycam_paths:
        # Only save original images to databases
        original_img_dir = skycam_paths[skycam_dir]['img_subdirs']['original']
        for original_fname in os.listdir(original_img_dir):
            if original_fname[-4:] == '.jpg':
                # Collect image features
                img_uid = get_uid(original_fname)
                skycam_type = original_fname.split('_')[0]
                t = get_img_time(original_fname)
                timestamp = (t - datetime(1970, 1, 1)) / timedelta(seconds=1)
                batch_data_subdir = skycam_dir#get_img_path(original_fname, img_type, skycam_dir)
                # Add entries to img_df
                img_df = add_skycam_img(img_df, original_fname, skycam_type, timestamp, batch_id, batch_data_subdir, verbose=verbose)
                #print(dfs)
    return img_df

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
    img_df = add_skycam_data_to_img_df(img_df, batch_id, skycam_paths, verbose=True)
    save_df(img_df, 'img', None, batch_id, task, False, batch_path)
    if do_zip:
        zip_batch(task, batch_id, force_recreate=True)


def zip_batch(task, batch_id, root='batch_data_zipfiles', force_recreate=False):
    batch_path = batch_path = make_batch_dir(task, batch_id)
    batch_dir = get_batch_dir(task, batch_id)
    batch_zip_name = f'{root}/{batch_dir}'
    if force_recreate or not os.path.exists(batch_zip_name + '.tar.gz'):
        print(f"Zipping {batch_dir}")
        os.makedirs(root, exist_ok=True)
        shutil.make_archive(batch_zip_name, 'gztar', batch_path)


if __name__ == '__main__':
    build_batch(task='cloud-detection', batch_id=0, do_zip=True)
