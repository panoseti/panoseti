#! /usr/bin/env python3
import os
import random
import shutil
from datetime import datetime, timedelta

from skycam_utils import get_batch_dir, make_skycam_paths_json, get_skycam_img_time, get_skycam_img_path
from labeling_utils import add_skycam_img, get_dataframe, save_df
from preprocess_skycam import preprocess_skycam_imgs
"""
batch_data/ file tree

./batch_data/
├─ task_cloud-detection.batch-id_N/
│  ├─ skycam_imgs/
│  ├─ pano_imgs/
│  ├─ skycam_path_index.json
│  ├─ pano_path_index.json
│  ├─ task_cloud-detection.batch-id_N.type_feature.csv
"""

batch_data_root_dir = 'batch_data'
skycam_imgs_root_dir = 'skycam_imgs'
pano_imgs_root_dir = 'skycam_imgs'


def make_batch_dir(task, batch_id):
    batch_path = batch_data_root_dir + '/' + get_batch_dir(task, batch_id)
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
]

def add_skycam_data_to_skycam_df(skycam_df, batch_id, skycam_paths, verbose):
    """Add entries for each skycam image to skycam_df """
    for skycam_dir in skycam_paths:
        # Only save original images to databases
        original_img_dir = skycam_paths[skycam_dir]['img_subdirs']['original']
        for original_fname in os.listdir(original_img_dir):
            if original_fname[-4:] == '.jpg':
                # Collect image features
                skycam_type = original_fname.split('_')[0]
                t = get_skycam_img_time(original_fname)
                timestamp = (t - datetime(1970, 1, 1)) / timedelta(seconds=1)
                # Add entries to skycam_df
                skycam_df = add_skycam_img(skycam_df, original_fname, skycam_type, timestamp, batch_id, skycam_dir, verbose=verbose)
    return skycam_df

def build_batch(task, batch_id, do_zip=False):
    batch_path = make_batch_dir(task, batch_id)
    skycam_root_path = f'{batch_path}/{skycam_root_dirname}'
    for sample in samples:
        preprocess_skycam_imgs(sample['skycam_type'],
                               sample['year'],
                               sample['month'],
                               sample['day'],
                               root=skycam_root_path,
                               verbose=True)
    skycam_paths = make_skycam_paths_json(batch_path)
    skycam_df = get_dataframe('skycam')
    skycam_df = add_skycam_data_to_skycam_df(skycam_df, batch_id, skycam_paths, verbose=True)
    save_df(skycam_df, 'skycam', None, batch_id, task, False, batch_path)
    if do_zip:
        zip_batch(task, batch_id, force_recreate=True)


def zip_batch(task, batch_id, root='batch_data_zipfiles', force_recreate=False):
    batch_path = make_batch_dir(task, batch_id)
    batch_dir = get_batch_dir(task, batch_id)
    batch_zip_name = f'{root}/{batch_dir}'
    if force_recreate or not os.path.exists(batch_zip_name + '.tar.gz'):
        print(f"Zipping {batch_dir}")
        os.makedirs(root, exist_ok=True)
        shutil.make_archive(batch_zip_name, 'gztar', batch_path)


if __name__ == '__main__':
    build_batch(task='cloud-detection', batch_id=0, do_zip=True)
