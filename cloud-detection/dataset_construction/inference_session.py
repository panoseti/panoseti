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
    data_labels_path = f'./{data_labels_fname}'
    # TODO: continue refactoring code.

    def __init__(self, batch_id, task='cloud-detection'):
        super().__init__(batch_id, batch_type='inference')
        self.name = 'INFERENCE'
        self.user_uid = get_uid(self.name)
        self.batch_labels_path = f'{training_batch_labels_root_dir}/{self.batch_dir}'

        # Unzip batched data, if it exists
        os.makedirs(training_batch_data_root_dir, exist_ok=True)
        os.makedirs(self.batch_labels_path, exist_ok=True)
        try:
            unpack_batch_data(training_batch_data_root_dir)
            with open(f'{self.batch_path}/{self.pano_path_index_fname}', 'r') as f:
                self.pano_paths = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find \x1b[31m{self.batch_dir}\x1b[0m\n"
                                    f"Try adding the zipped data batch file to the following directory:\n"
                                    f"\x1b[31m{os.path.abspath(training_batch_data_root_dir)}\x1b[0m")
        try:
            with open(self.data_labels_path, 'r') as f:
                self.labels = json.load(f)
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
                save_dir=self.batch_path
            )
            if df is None:
                raise ValueError(f"Dataframe for '{df_type}' missing in batch directory!")

            self.loaded_dfs_from_file[df_type] = True
        elif df_type in ['unlabeled', 'labeled']:
            # These dataframes must be initialized here because, for simplicity, we
            # don't collect the user-uid until the user supplies them in the Jupyter notebook.
            df = load_df(
                self.user_uid, self.batch_id, df_type, self.task, is_temp=True,
                save_dir=self.batch_labels_path
            )
            if df is not None:
                self.loaded_dfs_from_file[df_type] = True
            else:
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

    def pano_uid_to_data(self, pano_uid, img_type):
        run_dir = self.pano_df.loc[
            (self.pano_df.pano_uid == pano_uid), 'run_dir'
        ].iloc[0]
        ptree = PanoBatchDataFileTree(self.batch_id, self.batch_type, run_dir)
        fpath = ptree.get_pano_img_path(pano_uid, img_type)
        img = np.asarray(Image.open(fpath))
        return img

    def start(self, debug=False):
        """Labeling interface that displays an image and prompts user for its class."""
        data_to_label = self.unlabeled_df.loc[self.unlabeled_df.is_labeled == False]

        if len(data_to_label) == 0:
            emojis = ['🌈', '💯', '✨', '🎉', '🎃', '🔭', '🌌']
            print(f"All data are labeled! {np.random.choice(emojis)}")
            return
        try:
            i = data_to_label.index[0]
            while i < len(self.unlabeled_df):
                # Clear display then show next image to label
                feature_uid = self.unlabeled_df.iloc[i]['feature_uid']
                if debug:
                    label_str = np.random.choice(list(self.labels.values()))
                    self.labeled_df = add_labeled_data(self.labeled_df, self.unlabeled_df, feature_uid, self.user_uid, label_str)
                    i += 1
                    continue

                # Get image label from model

                # TODO
                label_val = ...
                label_str = self.labels[str(label_val)]
                self.labeled_df = add_labeled_data(self.labeled_df, self.unlabeled_df, feature_uid, self.user_uid, label_str)
                i += 1
        finally:
            self.save_progress()
            print('Success!')

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
