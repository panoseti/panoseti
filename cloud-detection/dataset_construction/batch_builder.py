#! /usr/bin/env python3
"""
Builds batches of unlabeled data for human labeling, referred to as "data batches" in the code.

The labeling architecture assumes that these data batches of data are the smallest
unit of data that is distributed to users. The lifecycle of a data batch is the following:
    - Data batches are produced by this script (NOTE: must be run on a server with access to all relevant panoseti data.)
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
import shutil

from batch_building_utils import (CloudDetectionBatchDataFileTree, SkycamBatchDataFileTree, PanoBatchDataFileTree,
                                  get_batch_def_json_fname, training_batch_data_zipfiles_dir,
                                  inference_batch_data_zipfiles_dir, valid_skycam_img_types)
from dataframe_utils import get_dataframe, save_df
from skycam_builder import SkycamBatchBuilder
from pano_builder import PanoBatchBuilder


class CloudDetectionBatchBuilder(CloudDetectionBatchDataFileTree):
    def __init__(self, batch_id, batch_def, batch_type, verbose=False, do_zip=True, prune_skycam=False, force_recreate=False, manual_skycam_download=False):
        super().__init__(batch_id, batch_type, root='../dataset_construction')
        self.batch_def = batch_def
        self.verbose = verbose
        self.do_zip = do_zip
        self.prune_skycam = prune_skycam
        self.force_recreate = force_recreate
        self.manual_skycam_download = manual_skycam_download

        self.feature_df = get_dataframe('feature')
        self.pano_df = get_dataframe('pano')
        self.skycam_df = get_dataframe('skycam')

        if force_recreate and os.path.exists(self.batch_path):
                # response = input(f'Force recreating will remove {os.path.abspath(self.batch_path)}.\n'
                #                  f'Enter [y] to proceed.')
                response = 'y'
                if response == 'y':
                    print(f'Removing {os.path.abspath(self.batch_path)}')
                    shutil.rmtree(self.batch_path)
                else:
                    print('Cancelling force recreate and building as normal')

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
        for pano_run_dir_name in os.listdir(self.pano_root_path):
            ptree = PanoBatchDataFileTree(self.batch_id, self.batch_type, pano_run_dir_name)
            if os.path.isdir(ptree.pano_path) and 'pffd' in pano_run_dir_name:
                pano_paths[ptree.pano_path] = {
                    "img_subdirs": {},
                    "imgs_per_subdir": -1,
                }
                pano_paths[ptree.pano_path]["img_subdirs"] = ptree.pano_subdirs
                num_imgs_per_subdir = []
                for img_type, subdir in ptree.pano_subdirs.items():
                    if img_type in ['original', 'derivative', 'fft', 'fft-derivative']:
                        print(json.dumps(pano_paths, indent=4))
                        continue
                    num_imgs_per_subdir.append(len(os.listdir(subdir)))
                if not all([e == num_imgs_per_subdir[0] for e in num_imgs_per_subdir]):
                    raise Warning(f"Unequal number of image in {ptree.pano_path}. Some pano img types are missing data.")
                pano_paths[ptree.pano_path]["imgs_per_subdir"] = num_imgs_per_subdir[0]
        with open(f"{self.batch_path}/{self.pano_path_index_fname}", 'w') as f:
            f.write(json.dumps(pano_paths, indent=4))
        return pano_paths

    def make_skycam_paths_json(self):
        """Create file for indexing sky-camera image paths."""
        assert os.path.exists(self.batch_path), f"Could not find the batch directory {self.batch_path}"
        skycam_paths = {}
        for skycam_dir in os.listdir(self.skycam_root_path):
            sctree = SkycamBatchDataFileTree(self.batch_id, self.batch_type, skycam_dir=skycam_dir)
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
        """Create a zipfile for the current data batch."""
        os.makedirs(training_batch_data_zipfiles_dir, exist_ok=True)
        if self.batch_type == 'training':
            batch_zip_name = f'{training_batch_data_zipfiles_dir}/{self.batch_dir}'
        elif self.batch_type == 'inference':
            batch_zip_name = f'{inference_batch_data_zipfiles_dir}/{self.batch_dir}'
        batch_zip_path = batch_zip_name + '.tar.gz'
        if self.force_recreate and os.path.exists(batch_zip_name + '.tar.gz'):
            os.remove(batch_zip_path)
        print(f"\nZipping {self.batch_dir}")
        shutil.make_archive(batch_zip_name, 'gztar', self.batch_path)
        print(f'\tAbs path: {os.path.abspath(batch_zip_path)}\n'
              f'\tSize: {round(os.path.getsize(batch_zip_path) / 2**20, 2)} MB')
        print("Done")

    def prune_skycam_imgs(self):
        skycam_uids_in_feature_df = self.feature_df.loc[:, 'skycam_uid']
        skycam_not_in_feature_df = self.skycam_df.loc[~self.skycam_df['skycam_uid'].isin(skycam_uids_in_feature_df)]
        for skycam_dir in skycam_not_in_feature_df['skycam_dir'].unique():
            sctree = SkycamBatchDataFileTree(self.batch_id, self.batch_type, skycam_dir=skycam_dir)
            unused_skycam_fnames = skycam_not_in_feature_df.loc[skycam_not_in_feature_df['skycam_dir'] == skycam_dir, 'fname']
            for fname in unused_skycam_fnames:
                for skycam_type in valid_skycam_img_types:
                    fpath = sctree.get_skycam_img_path(fname, skycam_type)
                    os.remove(fpath)
        self.skycam_df.drop(skycam_not_in_feature_df.index, inplace=True)
        self.skycam_df.reset_index(drop=True, inplace=True)

    def build_training_batch(self):
        for sample_dict in self.batch_def:
            skycam_builder = SkycamBatchBuilder(
                self.batch_id,
                self.batch_type,
                **sample_dict['skycam'],
                verbose=self.verbose,
                force_recreate=False
            )
            self.skycam_df = skycam_builder.build_skycam_batch_data(
                self.skycam_df,
                do_manual_skycam_download=self.manual_skycam_download
            )

        for sample_dict in self.batch_def:
            pano_builder = PanoBatchBuilder(
                'cloud-detection',
                self.batch_id,
                self.batch_type,
                sample_dict['pano']['data_dir'],
                sample_dict['pano']['run_dir'],
                verbose=self.verbose,
                force_recreate=self.force_recreate,
                do_baseline_subtraction=True,
            )
            sctree = SkycamBatchDataFileTree(self.batch_id, self.batch_type, **sample_dict['skycam'])
            self.feature_df, self.pano_df = pano_builder.build_pano_training_batch_data(
                self.feature_df, self.pano_df, self.skycam_df, sctree.skycam_dir, sample_dict['sample_stride'],
                upsample_pano_frames=int(sample_dict["upsample_pano_frames"])
            )

        try:
            save_df(
                self.feature_df, 'feature', None, self.batch_id,
                self.task, False, self.batch_path, overwrite_ok=self.force_recreate
            )
            save_df(
                self.pano_df, 'pano', None, self.batch_id,
                self.task, False, self.batch_path, overwrite_ok=self.force_recreate
            )
            if self.prune_skycam:
                self.prune_skycam_imgs()
            save_df(
                self.skycam_df, 'skycam', None, self.batch_id,
                self.task, False, self.batch_path, overwrite_ok=self.force_recreate
            )
        except FileExistsError as fee:
            print('Dataframes already created.')

        self.make_skycam_paths_json()
        self.make_pano_paths_json()
        self.make_batch_def_json()

        if self.do_zip:
            self.zip_batch()

    def build_inference_batch(self):
        for sample_dict in self.batch_def:
            pano_builder = PanoBatchBuilder(
                'cloud-detection',
                self.batch_id,
                self.batch_type,
                sample_dict['pano']['data_dir'],
                sample_dict['pano']['run_dir'],
                do_baseline_subtraction=True,
                verbose=self.verbose,
                force_recreate=self.force_recreate,
            )
            if 'time_step' not in sample_dict:
                sample_dict['time_step'] = 60
            self.feature_df, self.pano_df = pano_builder.build_pano_inference_batch_data(
                self.feature_df, self.pano_df, float(sample_dict['time_step'])
            )

        try:
            save_df(
                self.feature_df, 'feature', None, self.batch_id,
                self.task, False, self.batch_path, overwrite_ok=self.force_recreate
            )
            save_df(
                self.pano_df, 'pano', None, self.batch_id,
                self.task, False, self.batch_path, overwrite_ok=self.force_recreate
            )
            # if self.prune_skycam:
            #     self.prune_skycam_imgs()
            # save_df(
            #     self.skycam_df, 'skycam', None, self.batch_id,
            #     self.task, False, self.batch_path, overwrite_ok=self.force_recreate
            # )
        except FileExistsError as fee:
            print('Dataframes already created.')

        # self.make_skycam_paths_json()
        self.make_pano_paths_json()
        self.make_batch_def_json()

        if self.do_zip:
            self.zip_batch()
