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

from skycam_utils import make_skycam_paths_json
from batch_building_utils import *
from dataframe_utils import get_dataframe, save_df, load_df

from skycam_builder import SkycamBatchBuilder

from pano_utils import make_pano_paths_json
from pano_builder import PanoBatchBuilder

from dataset_manager import CloudDetectionDatasetManager
from base_classes import GenericDataBatchBuilder


class CloudDetectionBatchBuilder:
    def __init__(self, task, batch_id, batch_def, verbose, do_zip, force_recreate, manual_skycam_download):
        self.task = task
        self.batch_id = batch_id
        self.batch_def = batch_def
        self.verbose = verbose
        self.do_zip = do_zip
        self.force_recreate = force_recreate
        self.manual_skycam_download = manual_skycam_download

        self.batch_path = get_batch_path(task, batch_id)
        self.batch_dir = get_batch_dir(task, batch_id)

        self.feature_df = get_dataframe('feature')
        self.pano_df = get_dataframe('pano')
        self.skycam_df = get_dataframe('skycam')

    def init_data_batch_dir(self):
        os.makedirs(self.batch_path, exist_ok=True)

    def make_batch_def_json(self):
        batch_def_fname = get_batch_def_json_fname(self.task, self.batch_id)
        with open(f"{self.batch_path}/{batch_def_fname}", 'w') as f:
            f.write(json.dumps(self.batch_def, indent=4))


    def zip_batch(self):
        os.makedirs(batch_data_zipfiles_dir, exist_ok=True)
        batch_zip_name = f'{batch_data_zipfiles_dir}/{self.batch_dir}'
        if self.force_recreate or not os.path.exists(batch_zip_name + '.tar.gz'):
            if self.force_recreate and os.path.exists(batch_zip_name + '.tar.gz'):
                os.remove(batch_zip_name + '.tar.gz', )
            print(f"\nZipping {self.batch_dir}")
            shutil.make_archive(batch_zip_name, 'gztar', self.batch_path)
            print("Done")

    def prune_skycam_imgs(self):
        skycam_uids_in_feature_df = self.feature_df.loc[:, 'skycam_uid']
        skycam_not_in_feature_df = self.skycam_df.loc[~self.skycam_df['skycam_uid'].isin(skycam_uids_in_feature_df)]
        for skycam_dir in skycam_not_in_feature_df['skycam_dir'].unique():
            unused_skycam_fnames = skycam_not_in_feature_df.loc[skycam_not_in_feature_df['skycam_dir'] == skycam_dir, 'fname']
            skycam_path = get_skycam_path(self.batch_path, skycam_dir)
            for fname in unused_skycam_fnames:
                for skycam_type in valid_skycam_img_types:
                    fpath = get_skycam_img_path(fname, skycam_type, skycam_path)
                    os.remove(fpath)
        self.skycam_df.drop(skycam_not_in_feature_df.index, inplace=True)
        self.skycam_df.reset_index(drop=True, inplace=True)

    def build_batch(self):
        for sample_dict in self.batch_def:
            print(f'\nBuilding features for {sample_dict}')
            pano_builder = PanoBatchBuilder(
                sample_dict['pano']['data_dir'],
                sample_dict['pano']['run_dir'],
                'cloud-detection',
                self.batch_id,
                verbose=self.verbose,
                force_recreate=self.force_recreate,
            )

            skycam_builder = SkycamBatchBuilder(
                self.task,
                self.batch_id,
                self.batch_path,
                sample_dict['skycam']['skycam_type'],
                sample_dict['skycam']['year'],
                sample_dict['skycam']['month'],
                sample_dict['skycam']['day'],
                verbose=self.verbose,
                force_recreate=False
            )

            self.skycam_df = skycam_builder.build_skycam_batch_data(
                self.skycam_df,
                do_manual_skycam_download=self.manual_skycam_download
            )

            self.feature_df, self.pano_df = pano_builder.build_pano_batch_data(
                self.feature_df, self.pano_df, self.skycam_df, skycam_builder.skycam_dir
            )

        self.prune_skycam_imgs()

        try:
            save_df(
                self.feature_df, 'feature', None, self.batch_id,
                self.task, False, self.batch_path, overwrite_ok=self.force_recreate
            )
            save_df(
                self.skycam_df, 'skycam', None, self.batch_id,
                self.task, False, self.batch_path, overwrite_ok=self.force_recreate
            )
            save_df(
                self.pano_df, 'pano', None, self.batch_id,
                self.task, False, self.batch_path, overwrite_ok=self.force_recreate
            )
        except FileExistsError as fee:
            print('Dataframes already created.')

        make_skycam_paths_json(self.batch_path)
        make_pano_paths_json(self.batch_path)
        self.make_batch_def_json()

        if self.do_zip:
            self.zip_batch()



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
        {
            'pano': {
                'data_dir': DATA_DIR,
                'run_dir': 'obs_Lick.start_2023-08-29T04:49:58Z.runtype_sci-obs.pffd',
            },
            'skycam': {
                'skycam_type': 'SC2',
                'year': 2023,
                'month': 8,
                'day': 28
            }
        },
    ]
    batch_id = 6

    data_batch_builder = CloudDetectionBatchBuilder(
        'cloud-detection',
        batch_id,
        batch_def_3,
        verbose=True,
        do_zip=True,
        force_recreate=True,
        manual_skycam_download=False
    )
    data_batch_builder.build_batch()

    #zip_batch('cloud-detection', 4, force_recreate=True)
