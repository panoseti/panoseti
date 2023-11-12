#! /usr/bin/env python3
import os
import random
import shutil
from datetime import datetime, timedelta

from skycam_utils import *
from dataframe_utils import add_feature_entry, get_dataframe, save_df, batch_data_root_dir, pano_imgs_root_dir
from preprocess_skycam import preprocess_skycam_imgs

"""
Batched data file tree:
------------

batch_data/
├─ task_cloud-detection.batch-id_0/
├─ task_cloud-detection.batch-id_1/
.
.
.
├─ task_cloud-detection.batch-id_N/
│  ├─ skycam_imgs/
│  │  ├─ original/
│  │  ├─ cropped/
│  │  ├─ pfov/
│  ├─ pano_imgs/
│  │  ├─ original/
│  │  ├─ derivative/
│  │  ├─ fft/
│  ├─ skycam_path_index.json
│  ├─ pano_path_index.json
│  ├─ task_cloud-detection.batch-id_N.type_feature.csv
.
.
.
------------
"""

batch_data_zipfiles_dir = 'batch_data_zipfiles'


def make_batch_dir(task, batch_id):
    batch_path = batch_data_root_dir + '/' + get_batch_dir(task, batch_id)
    os.makedirs(batch_path, exist_ok=True)
    return batch_path

def zip_batch(task, batch_id, force_recreate=False):
    batch_path = make_batch_dir(task, batch_id)
    batch_dir = get_batch_dir(task, batch_id)
    batch_zip_name = f'{batch_data_zipfiles_dir}/{batch_dir}'
    if force_recreate or not os.path.exists(batch_zip_name + '.tar.gz'):
        print(f"Zipping {batch_dir}")
        os.makedirs(batch_data_zipfiles_dir, exist_ok=True)
        shutil.make_archive(batch_zip_name, 'gztar', batch_path)


def select_samples():
    ...

samples = [
    {
        'skycam_type': 'SC2',
        'year': 2023,
        'month': 7,
        'day': 31
    },
]



def create_skycam_df(batch_id, batch_path, first_t, last_t):
    skycam_imgs_root_path = f'{batch_path}/{skycam_imgs_root_dir}'
    skycam_df = get_dataframe('skycam')
    for sample in samples:
        preprocess_skycam_imgs(sample['skycam_type'],
                               sample['year'],
                               sample['month'],
                               sample['day'],
                               first_t,
                               last_t,
                               root=skycam_imgs_root_path,
                               verbose=True)
        skycam_dir = get_skycam_dir(sample['skycam_type'], sample['year'], sample['month'], sample['day'])
        skycam_df = add_skycam_data_to_skycam_df(skycam_df, batch_id, skycam_imgs_root_path, skycam_dir, verbose=True)

    skycam_paths = make_skycam_paths_json(batch_path)
    return skycam_df

def create_pano_df(batch_id, batch_path):
    pano_imgs_root_path = f'{batch_path}/{pano_imgs_root_dir}'
    pano_df = get_dataframe('pano')
    return pano_df


def create_feature_df(batch_id, batch_path, skycam_df, pano_df):
    feature_df = get_dataframe('feature')
    for i in range(len(skycam_df)):
        skycam_uid = skycam_df.iloc[i]['skycam_uid']
        feature_df = add_feature_entry(feature_df, skycam_uid, 'None', batch_id)
    return feature_df


def build_batch(task, batch_id, first_utc, last_utc, do_zip=False):
    batch_path = make_batch_dir(task, batch_id)

    skycam_df = create_skycam_df(batch_id, batch_path, first_utc, last_utc)
    pano_df = create_pano_df(batch_id, batch_path)
    feature_df = create_feature_df(batch_id, batch_path, skycam_df, pano_df)

    save_df(skycam_df, 'skycam', None, batch_id, task, False, batch_path)
    save_df(pano_df, 'pano', None, batch_id, task, False, batch_path)
    save_df(feature_df, 'feature', None, batch_id, task, False, batch_path)
    if do_zip:
        zip_batch(task, batch_id, force_recreate=True)



if __name__ == '__main__':
    build_batch(task='cloud-detection', batch_id=0, do_zip=True)
