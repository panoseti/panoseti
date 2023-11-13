#! /usr/bin/env python3
import os
import random
import shutil
from datetime import datetime, timedelta

from skycam_utils import *
from dataframe_utils import get_dataframe, save_df, batch_data_root_dir, pano_imgs_root_dir, skycam_imgs_root_dir, skycam_path_index_fname
from preprocess_skycam import preprocess_skycam_imgs
from panoseti_batch_utils import make_pano_paths_json
from panoseti_batch_builder import PanosetiBatchBuilder

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



def create_skycam_features(sample_dict, skycam_df, batch_id, batch_path, first_t, last_t):
    skycam_imgs_root_path = get_skycam_root_path(batch_path)
    skycam_info = sample_dict['skycam']
    skycam_dir = get_skycam_dir(
        skycam_info['skycam_type'], skycam_info['year'], skycam_info['month'], skycam_info['day']
    )
    preprocess_skycam_imgs(skycam_info['skycam_type'],
                           skycam_info['year'],
                           skycam_info['month'],
                           skycam_info['day'],
                           first_t,
                           last_t,
                           root=skycam_imgs_root_path,
                           verbose=True)
    skycam_df = add_skycam_data_to_skycam_df(skycam_df, batch_id, skycam_imgs_root_path, skycam_dir, verbose=True)
    return skycam_df

def create_pano_df(batch_id, batch_path):
    pano_imgs_root_path = f'{batch_path}/{pano_imgs_root_dir}'
    pano_df = get_dataframe('pano')
    return pano_df


def create_feature_df(batch_id, batch_path, skycam_df, pano_df):
    feature_df = get_dataframe('feature')
    return feature_df


def build_batch(sample_dict, task, batch_id, do_zip=False):
    batch_path = make_batch_dir(task, batch_id)

    skycam_df = get_dataframe('skycam')
    pano_df = get_dataframe('pano')
    feature_df = get_dataframe('feature')

    pano_builder = PanosetiBatchBuilder(
        sample_dict['pano']['data_dir'],
        sample_dict['pano']['run_dir'],
        'cloud-detection',
        batch_id,
        force_recreate=True
    )

    skycam_df = create_skycam_features(
        sample_dict, skycam_df, batch_id, batch_path, pano_builder.start_utc, pano_builder.stop_utc
    )

    skycam_dir = get_skycam_dir(
        sample_dict['skycam']['skycam_type'],
        sample_dict['skycam']['year'],
        sample_dict['skycam']['month'],
        sample_dict['skycam']['day']
    )

    feature_df, pano_df = pano_builder.create_feature_images(
        feature_df, pano_df, skycam_dir, verbose=True
    )

    save_df(skycam_df, 'skycam', None, batch_id, task, False, batch_path)
    save_df(pano_df, 'pano', None, batch_id, task, False, batch_path)
    save_df(feature_df, 'feature', None, batch_id, task, False, batch_path)

    skycam_paths = make_skycam_paths_json(batch_path)
    pano_paths = make_pano_paths_json(batch_path)
    if do_zip:
        zip_batch(task, batch_id, force_recreate=True)



if __name__ == '__main__':


    DATA_DIR = '/Users/nico/Downloads/panoseti_test_data/obs_data/data'
    # RUN_DIR = 'obs_Lick.start_2023-08-29T04:49:58Z.runtype_sci-obs.pffd'
    RUN_DIR = 'obs_Lick.start_2023-08-01T05:14:21Z.runtype_sci-obs.pffd'

    samples = [
        {
            'pano': {
                'data_dir': DATA_DIR,
                'run_dir': RUN_DIR,
            },
            'skycam': {
                'skycam_type': 'SC2',
                'year': 2023,
                'month': 7,
                'day': 31
            }
        },
    ]

    build_batch(samples[0], 'cloud-detection', 0, do_zip=True)
