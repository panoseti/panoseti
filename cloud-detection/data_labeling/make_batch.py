#! /usr/bin/env python3
"""
Builds batches of unlabeled data for human labeling, referred to as "data batches" in the code.

The labeling architecture assumes that these data batches of data are the smallest
unit of data that is distributed to users. The lifecycle of a data batch is the following:
    - Data batches are produced by this script, running on a server with access to all requisite source data (i.e. panoseti data).
    - These batches are uploaded to the cloud for distribution to users.
    - Users download data batches and label them using the PANOSETI Data Labeling Interface.
    - Users upload data batch labels to the cloud.
    - DatasetManager aggregates data batches and user labels into a labeled training dataset.

Specify how to build a batch with a samples.json file (TODO)
NOTE: to make coordinating labeling efforts simpler, batch definitions
should never change after being finalized and distributed to labelers.
Creating new batches and deleting older versions is the preferred way of
dealing with faulty batches.

Run build_batch to automatically generate a zipped batch.

Each data batch has the following file tree format:
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
│  │  ├─ fft/
│  │  ├─ derivative/
│  │  ├─ fft-derivative/
│  ├─ skycam_path_index.json
│  ├─ pano_path_index.json
│  ├─ task_cloud-detection.batch-id_N.type_feature.csv
│  ├─ task_cloud-detection.batch-id_N.type_pano.csv
│  ├─ task_cloud-detection.batch-id_N.type_skycam.csv
.
.
.
------------
"""

import os
import random
import shutil
from datetime import datetime, timedelta

from skycam_utils import *
from batch_building_utils import *
from dataframe_utils import get_dataframe, save_df
from preprocess_skycam import preprocess_skycam_imgs
from pano_utils import make_pano_paths_json
from pano_builder import PanosetiBatchBuilder
from dataset_manager import DatasetManager


def make_batch_def_json(batch_path, task, batch_id, batch_def):
    batch_def_fname = get_batch_def_json_fname(task, batch_id)
    with open(f"{batch_path}/{batch_def_fname}", 'w') as f:
        f.write(json.dumps(batch_def, indent=4))


def zip_batch(task, batch_id, force_recreate=True):
    os.makedirs(batch_data_zipfiles_dir, exist_ok=True)
    batch_path = get_batch_path(task, batch_id)
    batch_dir = get_batch_dir(task, batch_id)
    batch_zip_name = f'{batch_data_zipfiles_dir}/{batch_dir}'
    if force_recreate or not os.path.exists(batch_zip_name + '.tar.gz'):
        if force_recreate and os.path.exists(batch_zip_name + '.tar.gz'):
            os.remove(batch_zip_name + '.tar.gz', )
        print(f"\nZipping {batch_dir}")
        shutil.make_archive(batch_zip_name, 'gztar', batch_path)
        print("Done")


def create_skycam_features(sample_dict,
                           skycam_df,
                           batch_id,
                           batch_path,
                           first_t,
                           last_t,
                           manual_skycam_download,
                           verbose):
    skycam_info = sample_dict['skycam']
    skycam_dir = get_skycam_dir(
        skycam_info['skycam_type'], skycam_info['year'], skycam_info['month'], skycam_info['day']
    )
    print(f'Creating skycam features for {skycam_dir}')
    preprocess_skycam_imgs(skycam_info['skycam_type'],
                           skycam_info['year'],
                           skycam_info['month'],
                           skycam_info['day'],
                           first_t,
                           last_t,
                           batch_path,
                           manual_skycam_download,
                           verbose=verbose)
    skycam_df = add_skycam_data_to_skycam_df(
        skycam_df, batch_id, get_skycam_root_path(batch_path), skycam_dir, verbose=verbose
    )
    return skycam_df

def build_batch(batch_def,
                task,
                batch_id,
                verbose=False,
                do_zip=False,
                force_recreate=False,
                manual_skycam_download=False):
    batch_path = get_batch_path(task, batch_id)
    os.makedirs(batch_path, exist_ok=True)

    skycam_df = get_dataframe('skycam')
    pano_df = get_dataframe('pano')
    feature_df = get_dataframe('feature')


    for sample_dict in batch_def:
        print(f'\nBuilding features for {sample_dict}')
        pano_builder = PanosetiBatchBuilder(
            sample_dict['pano']['data_dir'],
            sample_dict['pano']['run_dir'],
            'cloud-detection',
            batch_id,
            force_recreate=force_recreate
        )

        skycam_df = create_skycam_features(
            sample_dict, skycam_df, batch_id, batch_path, pano_builder.start_utc, pano_builder.stop_utc,
            manual_skycam_download=manual_skycam_download, verbose=verbose
        )

        skycam_dir = get_skycam_dir(
            sample_dict['skycam']['skycam_type'],
            sample_dict['skycam']['year'],
            sample_dict['skycam']['month'],
            sample_dict['skycam']['day']
        )

        try:
            pano_builder.init_preprocessing_dirs()
        except FileExistsError:
            if not pano_builder.force_recreate:
                print(f'Data in {pano_builder.pano_path} already processed')
                continue

        print(f'Creating panoseti features for {sample_dict["pano"]["run_dir"]}')
        for module_id in [1]:
            feature_df, pano_df = pano_builder.create_feature_images(
                feature_df, pano_df, skycam_dir, module_id, verbose=verbose
            )
    try:
        save_df(skycam_df, 'skycam', None, batch_id, task, False, batch_path, overwrite_ok=False)
        save_df(pano_df, 'pano', None, batch_id, task, False, batch_path, overwrite_ok=False)
        save_df(feature_df, 'feature', None, batch_id, task, False, batch_path, overwrite_ok=False)
        dataset_manager.aggregate_batch_data_features(batch_id)
    except FileExistsError as fee:
        print('Dataframes already created.')

    make_skycam_paths_json(batch_path)
    make_pano_paths_json(batch_path)
    make_batch_def_json(batch_path, task, batch_id, batch_def)

    if do_zip:
        zip_batch(task, batch_id, force_recreate=True)


if __name__ == '__main__':
    DATA_DIR = '/Users/nico/Downloads/panoseti_test_data/obs_data/data'

    batch_def_3 = [
        {
            'pano': {
                'data_dir': DATA_DIR,
                'run_dir': 'obs_Lick.start_2023-08-01T05:14:21Z.runtype_sci-obs.pffd',
            },
            'skycam': {
                'skycam_type': 'SC2',
                'year': 2023,
                'month': 7,
                'day': 31
            }
        },
        # {
        #     'pano': {
        #         'data_dir': DATA_DIR,
        #         'run_dir': 'obs_Lick.start_2023-08-24T04:37:00Z.runtype_sci-obs.pffd',
        #     },
        #     'skycam': {
        #         'skycam_type': 'SC2',
        #         'year': 2023,
        #         'month': 8,
        #         'day': 23
        #     }
        # },
        # {
        #     'pano': {
        #         'data_dir': DATA_DIR,
        #         'run_dir': 'obs_Lick.start_2023-08-29T04:49:58Z.runtype_sci-obs.pffd',
        #     },
        #     'skycam': {
        #         'skycam_type': 'SC2',
        #         'year': 2023,
        #         'month': 8,
        #         'day': 28
        #     }
        # },
    ]
    batch_id = 2

    dataset_manager = DatasetManager()
    build_batch(batch_def_3,
                'cloud-detection',
                batch_id,
                verbose=True,
                do_zip=True,
                force_recreate=False,
                manual_skycam_download=False)
    #zip_batch('cloud-detection', 4, force_recreate=True)
