#! /usr/bin/env python3
"""
Labeling interface for model inference
"""
import os
import sys
import shutil
import json
import math
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

from batch_building_utils import *
from dataframe_utils import *

class InferenceSession(CloudDetectionBatchDataFileTree):
    dataset_root = '../dataset_construction'
    data_labels_path = f'{dataset_root}/{data_labels_fname}'
    # TODO: continue refactoring code.

    def __init__(self, batch_id, task='cloud-detection'):
        super().__init__(batch_id, batch_type='inference')
        self.name = 'INFERENCE'
        self.user_uid = get_uid(self.name)
        self.batch_labels_path = f'{self.dataset_root}/{inference_batch_labels_root_dir}/{self.batch_dir}'

        # Unzip batched data, if it exists
        os.makedirs(self.batch_labels_path, exist_ok=True)
        try:
            unpack_batch_data(f'{self.dataset_root}/{inference_batch_data_root_dir}')
            with open(f'{self.dataset_root}/{self.batch_path}/{self.pano_path_index_fname}', 'r') as f:
                self.pano_paths = json.load(f)
        except FileNotFoundError as e:
            raise e
            # raise FileNotFoundError(f"Could not find \x1b[31m{self.batch_dir}\x1b[0m\n"
            #                         f"Try adding the zipped data batch file to the following directory:\n"
            #                         f"\x1b[31m{os.path.abspath(training_batch_data_root_dir)}\x1b[0m")
        try:
            with open(self.data_labels_path, 'r') as f:
                labels = json.load(f)
                self.labels = dict()
                for key in labels:
                    self.labels[int(key)] = labels[key]
        except:
            raise FileNotFoundError(f"Could not find the file {self.data_labels_path}\n"
                                    f"Try pulling the file from the panoseti github.")
        # Init dataframes
        self.loaded_dfs_from_file = {}
        self.feature_df = self.init_dataframe('feature')
        self.pano_df = self.init_dataframe('pano')
        self.unlabeled_df = self.init_dataframe('unlabeled')
        self.labeled_df = self.init_dataframe('labeled')


    def init_dataframe(self, df_type):
        """Attempt to load the given dataframe from file, if it exists. Otherwise, create a new dataframe."""
        if df_type in ['feature', 'pano']:
            # These dataframes are initialized by make_batch.py and should exist within the downloaded batch data.
            df = load_df(
                None, self.batch_id, df_type, self.task, is_temp=False,
                save_dir=self.dataset_root + '/' + self.batch_path
            )
            if df is None:
                raise ValueError(f"Dataframe for '{df_type}' missing in batch directory!")

            self.loaded_dfs_from_file[df_type] = True
        elif df_type in ['unlabeled', 'labeled']:
            # df = load_df(
            #     self.user_uid, self.batch_id, df_type, self.task, is_temp=True,
            #     save_dir=self.dataset_root + '/' + self.batch_labels_path
            # )
            # if df is not None:
            #     self.loaded_dfs_from_file[df_type] = True
            # else:
            self.loaded_dfs_from_file[df_type] = False
            df = get_dataframe(df_type)
            if df_type == 'unlabeled':
                # Note: must initialize feature_df before attempting to initialize unlabeled_df
                for feature_uid in self.feature_df['feature_uid']:
                    # Add entries to unlabeled_df
                    df = add_unlabeled_data(df, feature_uid)
        else:
            raise ValueError(f'Unsupported df_type: "{df_type}"')
        return df

    def get_pano_feature_fpath(self, feature_uid, img_type):
        assert img_type in valid_pano_img_types, f"Image type '{img_type}' is not supported!"
        pano_uid = self.feature_df.loc[self.feature_df['feature_uid'] == feature_uid, 'pano_uid'].iloc[0]
        run_dir, batch_id = self.pano_df.loc[self.pano_df['pano_uid'] == pano_uid, ['run_dir', 'batch_id']].iloc[0]
        ptree = PanoBatchDataFileTree(batch_id, self.batch_type, run_dir)
        return ptree.get_pano_img_path(pano_uid, img_type)

    def pano_uid_to_data(self, pano_uid, img_type):
        run_dir = self.pano_df.loc[
            (self.pano_df.pano_uid == pano_uid), 'run_dir'
        ].iloc[0]
        ptree = PanoBatchDataFileTree(self.batch_id, self.batch_type, run_dir)
        fpath = ptree.get_pano_img_path(pano_uid, img_type)
        img = np.asarray(Image.open(fpath))
        return img

    def add_labels(self, inferences):
        assert len(inferences) == len(self.unlabeled_df)
        self.labeled_df['feature_uid'] = self.unlabeled_df['feature_uid']
        self.labeled_df['label'] = pd.Series(inferences).map(self.labels)
        self.labeled_df['user_uid'] = 'INFERENCE'
        self.unlabeled_df['is_labeled'] = True


    """State IO"""

    def save_progress(self):
        save_df(self.labeled_df,
                'labeled',
                self.user_uid,
                self.batch_id,
                self.task,
                True,
                self.batch_labels_path
                )

        save_df(self.unlabeled_df,
                'unlabeled',
                self.user_uid,
                self.batch_id,
                self.task,
                True,
                self.batch_labels_path
                )

    def create_export_zipfile(self, allow_partial_batch=False):
        """Create an export zipfile for the data only if all data has been labeled."""
        data_to_label = self.unlabeled_df[self.unlabeled_df.is_labeled == False]
        if not allow_partial_batch and len(data_to_label) > 0:
            print(f'Please label all data in the batch before exporting.')
            return
        print('Zipping batch labels...')
        user_label_export_dir = get_user_label_export_dir(self.task, self.batch_id, self.batch_type, self.user_uid, root='.')
        os.makedirs(user_label_export_dir, exist_ok=True)

        save_df(self.labeled_df,
                'labeled',
                self.user_uid,
                self.batch_id,
                self.task,
                False,
                user_label_export_dir
                )

        save_df(self.unlabeled_df,
                'unlabeled',
                self.user_uid,
                self.batch_id,
                self.task,
                False,
                user_label_export_dir
                )

        # Save user info
        user_info = {
            'name': self.name,
            'user-uid': self.user_uid
        }
        user_info_path = user_label_export_dir + '/' + 'user_info.json'
        with open(user_info_path, 'w') as f:
            f.write(json.dumps(user_info))

        shutil.make_archive(user_label_export_dir, 'zip', user_label_export_dir)
        shutil.rmtree(user_label_export_dir)
        print('Done!')

if __name__ == '__main__':
    InferenceSession(10)