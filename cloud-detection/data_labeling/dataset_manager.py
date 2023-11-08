#! /usr/bin/env python3
import sys
import os
import json
import shutil

import pandas as pd

from labeling_utils import get_dataframe, load_df, save_df, add_user, add_user_batch_log

sys.path.append('../../util')
#from pff import parse_name

class DatasetManager:
    label_log_fname = 'label_log.json'
    user_labeled_dir = 'user_labeled_batches'

    def __init__(self, task='cloud-detection'):
        self.task = task
        self.dataset_dir = f'dataset_{task}'
        self.label_log_path = f'{self.dataset_dir}/{self.label_log_fname}'
        self.user_labeled_path = f'{self.dataset_dir}/{self.user_labeled_dir}'

        #self.label_log = {}
        self.main_dfs = {   # Aggregated metadata datasets.
            'user': None,
            'img': None,
            'labeled': None,
            'user-batch-log': None
        }

        if not os.path.exists(self.dataset_dir):
            self.init_dataset_dir()
        else:
            #self.label_log = self.load_label_log()
            self.init_main_dfs()

    def init_dataset_dir(self):
        """
        Initialize dataset directory only if it does not exist
            - Create label_log file
            - Init dataframes
        """
        # Make dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=False)
        os.makedirs(self.user_labeled_path, exist_ok=False)
        # Make label_log file
        # with open(self.label_log_path, 'w') as f:
        #     self.label_log = {
        #         'batches': {
        #             0: {
        #                 'user-uids': []  # Records which users have labeled this batch
        #             }
        #         }
        #     }
        #     json.dump(self.label_log, f, indent=4)
        # Make empty dataframes from standard format function
        for df_type in self.main_dfs:
            self.main_dfs[df_type] = get_dataframe(df_type)  # Create df using standard definition
            self.save_main_df(df_type)

    @staticmethod
    def get_main_df_save_name(df_type):
        """Get standard main_df filename."""
        save_name = "type_{0}".format(df_type)
        return save_name + '.csv'

    @staticmethod
    def parse_name(name):
        d = {}
        x = name.split('.')
        for s in x:
            y = s.split('_')
            if len(y) < 2:
                continue
            d[y[0]] = y[1]
        return d


    def init_main_dfs(self):
        """Initialize the aggregated dataframes for the main dataset."""
        for df_type in self.main_dfs:
            self.main_dfs[df_type] = self.load_main_df(df_type)


    def load_main_df(self, df_type):
        """Load the main dataset dataframes."""
        df_path = f'{self.dataset_dir}/{self.get_main_df_save_name(df_type)}'
        if not os.path.exists(df_path):
            raise FileNotFoundError(f'{df_path} does not exist!')
        with open(df_path, 'r') as f:
            df = pd.read_csv(f, index_col=0)
            return df

    def save_main_df(self, df_type, overwrite_ok=True):
        """Save the main dataset dataframes."""
        df_path = f'{self.dataset_dir}/{self.get_main_df_save_name(df_type)}'
        df = self.main_dfs[df_type]
        if os.path.exists(df_path) and not overwrite_ok:
            raise FileNotFoundError(f'{df_path} exists and overwrite_ok=False. Aborting save...')
        else:
            with open(df_path, 'w'):
                df.to_csv(df_path)

    def load_label_log(self):
        """
        Load / create the label_log file. This file tracks which
        user-labeled data batches have been added to the master dataset.
        """
        if not os.path.exists(self.label_log_path):
            raise FileNotFoundError(f'{self.label_log_path} does not exist!')
        with open(self.label_log_path, 'r') as f:
            label_log = json.load(f)
        return label_log

    def save_label_log(self, label_log, overwrite_ok=True):
        """Save the label_log file."""
        if os.path.exists(self.label_log_path) and not overwrite_ok:
            raise FileNotFoundError(f'{self.label_log_path} exists and overwrite_ok=False. Aborting save...')
        else:
            with open(self.label_log_path, 'w') as f:
                json.dump(label_log, f, indent=4)

    def unpack_user_labeled_batches(self):
        for batch_name in os.listdir(self.user_labeled_path):
            if batch_name.endswith('.zip'):
                shutil.unpack_archive(f'{self.user_labeled_path}/{batch_name}', format='zip')


    def get_labeled_batch_info(self):
        """Return list of available user-labeled data batches."""
        labeled_batches = []
        if os.path.exists(self.user_labeled_path):
            print(self.user_labeled_path)
            self.unpack_user_labeled_batches()
            for batch_name in os.listdir(self.user_labeled_path):
                if not batch_name.startswith('task_'):
                    continue
                parsed = self.parse_name(batch_name)
                if parsed['task'] != self.task:
                    raise ValueError(f"Found data with a task={parsed['task']}, "
                                     f"which does not match the task of this dataset: {self.task}")
                labeled_batches.append(batch_name)
        return labeled_batches


    def update_dataset(self):
        """Incorporate each new user-labeled data batch into the dataset."""
        labeled_batches = self.get_labeled_batch_info()
        ubl_df = self.main_dfs['user-batch-log']
        user_df = self.main_dfs['user']
        for batch_name in labeled_batches:
            parsed = self.parse_name(batch_name)
            user_uid, batch_id = parsed['user-uid'], parsed['batch-id']

            batches_labeled_by_user = ubl_df.loc[
                ubl_df['user_uid'] == user_uid, 'batch_id'
            ]
            if batch_id not in batches_labeled_by_user:

                add_user_batch_log(ubl_df, user_uid, batch_id)
                # If user not in user_df, add them
                if user_uid not in user_df['user_uid']:
                    user_info_fname = f"{self.user_labeled_path}/{batch_name}/user_info.json"
                    with open(user_info_fname, "r") as f:
                        user_info = json.load(f)
                        add_user(user_df, user_uid, user_info['name'])
                # If batch
                # TODO: generate image_df and unlabeled_df definitions in the databatch
                # Add all labeled data
                user_labeled_df = load_df(
                    user_uid, batch_id, 'labeled', task=self.task,
                    is_temp=False, root=self.user_labeled_path, from_export=True
                )
                print(user_labeled_df)
                # self.save_main_df('user')
                # self.save_main_df('user-batch-log')



test = DatasetManager()
test.update_dataset()

