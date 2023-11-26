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

"""

import os
import json
import random
import shutil
from datetime import datetime, timedelta

from batch_building_utils import CloudDetectionBatchDataFileTree, SkycamBatchDataFileTree, PanoBatchDataFileTree, get_batch_def_json_fname, get_skycam_img_path, batch_data_zipfiles_dir, valid_skycam_img_types
from dataframe_utils import get_dataframe, save_df
from skycam_builder import SkycamBatchBuilder
from pano_builder import PanoBatchBuilder


class CloudDetectionBatchBuilder(CloudDetectionBatchDataFileTree):
    def __init__(self, task, batch_id, batch_def, verbose, do_zip, force_recreate, manual_skycam_download):
        super().__init__(task, batch_id)
        self.batch_def = batch_def
        self.verbose = verbose
        self.do_zip = do_zip
        self.force_recreate = force_recreate
        self.manual_skycam_download = manual_skycam_download

        self.feature_df = get_dataframe('feature')
        self.pano_df = get_dataframe('pano')
        self.skycam_df = get_dataframe('skycam')

    def init_data_batch_dir(self):
        os.makedirs(self.batch_path, exist_ok=True)

    def make_batch_def_json(self):
        batch_def_fname = get_batch_def_json_fname(self.task, self.batch_id)
        with open(f"{self.batch_path}/{batch_def_fname}", 'w') as f:
            f.write(json.dumps(self.batch_def, indent=4))

    def make_pano_paths_json(self):
        """Create file for indexing sky-camera image paths."""
        assert os.path.exists(self.batch_path), f"Could not find the batch directory {self.batch_path}"
        pano_paths = {}
        # pano_root_path = get_pano_root_path(self.batch_path)
        for path in os.listdir(self.pano_root_path):
            ptree = PanoBatchDataFileTree(self.task, self.batch_id, path)
            # pano_path = f'{self.pano_root_path}/{path}'
            if os.path.isdir(ptree.pano_path) and 'pffd' in path:
                pano_paths[ptree.pano_path] = {
                    "img_subdirs": {},
                    "imgs_per_subdir": -1,
                }
                # pano_subdirs = get_pano_subdirs(ptree.pano_path)
                pano_paths[ptree.pano_path]["img_subdirs"] = ptree.pano_subdirs
                num_imgs_per_subdir = []
                for subdir in ptree.pano_subdirs.values():
                    num_imgs_per_subdir.append(len(os.listdir(subdir)))
                if not all([e == num_imgs_per_subdir[0] for e in num_imgs_per_subdir]):
                    raise Warning(f"Unequal number of images in {ptree.pano_path}")
                pano_paths[ptree.pano_path]["imgs_per_subdir"] = num_imgs_per_subdir[0]
        with open(f"{self.batch_path}/{self.pano_path_index_fname}", 'w') as f:
            f.write(json.dumps(pano_paths, indent=4))
        return pano_paths

    def make_skycam_paths_json(self):
        """Create file for indexing sky-camera image paths."""
        assert os.path.exists(self.batch_path), f"Could not find the batch directory {self.batch_path}"
        skycam_paths = {}
        for skycam_dir in os.listdir(self.skycam_root_path):
            sctree = SkycamBatchDataFileTree(self.task, self.batch_id, skycam_dir)
            # skycam_path = f'{self.skycam_root_path}/{skycam_dir}'
            if os.path.isdir(sctree.skycam_path) and 'SC' in skycam_dir and 'imgs' in skycam_dir:
                skycam_paths[sctree.skycam_path] = {
                    "img_subdirs": {},
                    "imgs_per_subdir": -1,
                }
                skycam_paths[sctree.skycam_path]["img_subdirs"] = sctree.skycam_subdirs
                num_imgs_per_subdir = []
                for subdir in sctree.skycam_subdirs.values():
                    if subdir.endswith('original'):
                        continue
                    num_imgs_per_subdir.append(len(os.listdir(subdir)))
                if not all([e == num_imgs_per_subdir[0] for e in num_imgs_per_subdir]):
                    raise Warning(f"Unequal number of images in {sctree.skycam_path}")
                skycam_paths[sctree.skycam_path]["imgs_per_subdir"] = num_imgs_per_subdir[0]
        with open(f"{self.batch_path}/{self.skycam_path_index_fname}", 'w') as f:
            f.write(json.dumps(skycam_paths, indent=4))
        return skycam_paths

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
            sctree = SkycamBatchDataFileTree(self.task, self.batch_id, skycam_dir)
            unused_skycam_fnames = skycam_not_in_feature_df.loc[skycam_not_in_feature_df['skycam_dir'] == skycam_dir, 'fname']
            # skycam_path = get_skycam_path(self.batch_path, skycam_dir)
            for fname in unused_skycam_fnames:
                for skycam_type in valid_skycam_img_types:
                    fpath = get_skycam_img_path(fname, skycam_type, sctree.skycam_path)
                    os.remove(fpath)
        self.skycam_df.drop(skycam_not_in_feature_df.index, inplace=True)
        self.skycam_df.reset_index(drop=True, inplace=True)

    def build_batch(self):
        for sample_dict in self.batch_def:
            print(f'\nBuilding features for {sample_dict}')
            pano_builder = PanoBatchBuilder(
                'cloud-detection',
                self.batch_id,
                sample_dict['pano']['data_dir'],
                sample_dict['pano']['run_dir'],
                verbose=self.verbose,
                force_recreate=self.force_recreate,
            )

            skycam_builder = SkycamBatchBuilder(
                self.task,
                self.batch_id,
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

        self.make_skycam_paths_json()
        self.make_pano_paths_json()
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
    batch_id = 8

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
