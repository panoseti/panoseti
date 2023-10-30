import os
import shutil
import json
import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from PIL import Image
from IPython import display
import warnings

from skycam_utils import get_img_path, get_img_time, get_batch_dir
from labeling_utils import get_uid, get_batch_label_dir, get_dataframe, get_data_export_dir, add_labeled_data, add_unlabeled_data, add_skycam_img, save_df, load_df

class LabelSession:
    data_labels_file = 'skycam_labels.json'
    root_data_batch_dir = 'batch_data'
    img_paths_info_file = 'img_path_info.json'
    root_labeled_data_dir = 'batch_labels'

    def __init__(self, name, batch_id, task='cloud-detection'):
        self.name = name
        if name == "YOUR NAME":
            raise ValueError(f"Please enter your full name")
        self.user_uid = get_uid(name)

        self.batch_id = batch_id
        self.task = task

        self.batch_dir = get_batch_dir(task, batch_id)
        os.makedirs(LabelSession.root_data_batch_dir, exist_ok=True)
        os.makedirs(LabelSession.root_labeled_data_dir, exist_ok=True)
        self.batch_path = LabelSession.root_data_batch_dir + '/' + self.batch_dir

        self.loaded_dfs_from_file = False
        self.img_df = self.init_dataframe('img')
        self.unlabeled_df = self.init_dataframe('unlabeled-data')
        self.unlabeled_df = self.unlabeled_df.sample(frac=1).reset_index(drop=True)
        self.labeled_df = self.init_dataframe('labeled-data')
        self.data_to_label = None

        self.skycam_paths = {}
        self.img_uid_to_skycam_dir = {}
        self.init_skycam_paths()
        self.num_imgs = len(self.unlabeled_df)
        self.num_labeled = len(self.unlabeled_df.loc[self.unlabeled_df.is_labeled == True])

        with open(LabelSession.data_labels_file, 'r') as f:
            self.labels = json.load(f)

    def init_skycam_paths(self):
        try:
            with open(f'{self.batch_path}/{self.img_paths_info_file}', 'r') as f:
                self.skycam_paths = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find \x1b[31m{self.batch_dir}\x1b[0m\n"
                                    f"Try adding the (unzipped) data folder to the following directory:\n"
                                    f"\x1b[31m{os.path.abspath(self.root_data_batch_dir)}\x1b[0m")
        for skycam_dir in self.skycam_paths:
            self.init_img_uid_to_skycam_dir(skycam_dir)
            # Populate dataframes if they have not been loaded from file.
            if not self.loaded_dfs_from_file:
                self.init_skycam_df(skycam_dir)

    def init_skycam_df(self, skycam_dir):
        """Initialize img_df, unlabeled_df, and index (img_uid:skycam_dir) relations"""
        # Only save original images to databases
        original_img_dir = self.skycam_paths[skycam_dir]['img_subdirs']['original']
        for fname in os.listdir(original_img_dir):
            if fname[-4:] == '.jpg':
                # Collect image features
                img_uid = get_uid(fname)
                skycam_type = fname.split('_')[0]
                t = get_img_time(fname)
                timestamp = (t - datetime(1970, 1, 1)) / timedelta(seconds=1)
                # Add entries to img_df and unlabeled_df
                add_skycam_img(self.img_df, fname, skycam_type, timestamp)
                add_unlabeled_data(self.unlabeled_df, img_uid)

    def init_img_uid_to_skycam_dir(self, skycam_dir):
        """Save uid -> path relation for fast lookup later."""
        original_img_dir = self.skycam_paths[skycam_dir]['img_subdirs']['original']
        for fname in os.listdir(original_img_dir):
            if fname[-4:] == '.jpg':
                img_uid = get_uid(fname)
                self.img_uid_to_skycam_dir[img_uid] = skycam_dir

    def init_dataframe(self, df_type):
        """Attempt to load the given dataframe from file, if it exists. Otherwise, create a new dataframe."""
        df = load_df(self.user_uid, self.batch_id, df_type, self.task, is_temp=True)
        if df is None:
            df = get_dataframe(df_type)
        else:
            self.loaded_dfs_from_file = True
        return df


    def img_uid_to_data(self, img_uid, img_type):
        original_fname = self.img_df.loc[self.img_df.img_uid == img_uid, 'fname'].iloc[0]
        skycam_dir = self.img_uid_to_skycam_dir[img_uid]
        fpath = get_img_path(original_fname, img_type, skycam_dir)
        img = np.asarray(Image.open(fpath))
        return img

    # Plotting routines
    def plot_img(self, img_uid):
        """Given an image uid, display the corresponding image."""
        fig = plt.figure(figsize=(12, 12))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(1, 2),  # creates width x height grid of axes
                         axes_pad=0.3,  # pad between axes in inch.
                         share_all=False,
                         )
        imgs = [self.img_uid_to_data(img_uid, 'cropped'),
                self.img_uid_to_data(img_uid, 'pfov')]
        titles = [f'{img_uid[:8]}: cropped fov',
                  f'{img_uid[:8]}: full img with pfov']

        for i, (ax, img, title) in enumerate(zip(grid, imgs, titles)):
            ax.imshow(img, aspect='equal')
            ax.set_title(title)
        return fig

    def show_classifications(self):
        """Display all labeled images, organized by assigned class."""
        for key in self.labels.keys():
            self.make_img_grid(self.labeled_df, self.img_df, self.batch_dir, int(key))


    def make_img_grid(self, labeled_data_df, skycam_df, skycam_dir, label, cols=8, rows_per_plot=8):
        """Grid of all classified images labeled as the given label"""
        data_with_given_label = self.labeled_df.loc[(self.labeled_df.label == label), 'img_uid']
        imgs = [self.img_uid_to_data(img_uid, 'cropped') for img_uid in data_with_given_label]
        if len(imgs) == 0:
            print(f'No images labeled as "{self.labels[str(label)]}"')
            return
        else:
            print(f'Images you classified as "{self.labels[str(label)]}":')
        # Limit num rows in plot to ensure consistently-sized figures
        rows = math.ceil(len(imgs) / cols)
        num_subplots = rows_per_plot * cols
        for plot_idx in range(math.ceil(rows / rows_per_plot)):
            fig = plt.figure(figsize=(3. * rows_per_plot, 3. * cols))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                             nrows_ncols=(rows_per_plot, cols),  # creates width x height grid of axes
                             axes_pad=0.3,  # pad between axes in inch.
                             )
            for i in range(num_subplots):
                img_idx = plot_idx * num_subplots + i
                if img_idx < len(imgs):
                    ax = grid[i]
                    img = imgs[img_idx]
                    img_uid = data_with_given_label.iloc[img_idx]
                    ax.set_title(f'{23}{img_uid[:6]}')  # Label each plot with first 6 chars of img_uid
                    ax.imshow(img)
            plt.show()
            plt.close()

    # Labeling interface functions

    def get_valid_labels_str(self):
        """Returns: string describing all valid labels and their encodings."""
        s = ""
        for key, val in self.labels.items():
            s += f"{int(key) + 1}='{val}', "
        return s[:-2]

    def get_progress_str(self, num_symbols=50):
        """Generate status progress bar"""
        num_divisions = min(num_symbols, self.num_imgs)
        itrs_per_symbol = num_symbols / self.num_imgs
        prog = int(self.num_labeled * itrs_per_symbol)
        percent_labeled = self.num_labeled / self.num_imgs * 100
        progress_bar = '\u2593' * prog + '\u2591' * (num_symbols - prog)    # red text
        msg = f"|{progress_bar}|" + f"\t{self.num_labeled}/{self.num_imgs} ({percent_labeled:.2f}%)\n"
        return msg

    def get_user_label(self):
        """Prompts and returns the label a user assigns to a given image."""
        valid_label_str = self.get_valid_labels_str()
        progress_str = self.get_progress_str()
        valid_label = False
        is_first_itr = True
        while not valid_label:
            if is_first_itr:
                prompt = f"{progress_str}" \
                         f"Valid labels:\t{valid_label_str}" \
                         "\nYour label: "
            else:
                prompt = f"Valid labels:\t{valid_label_str}" \
                         "\nYour label: "
            label = input(prompt)
            if label.isnumeric() and str(int(label) - 1) in self.labels:
                valid_label = True
                label = int(label) - 1
            elif label == 'exit':
                return 'exit'
            else:
                first_itr = False
                print(f"\x1b[31mError:\t   '{label}' is not a valid label. "
                      "(To exit the session, type 'exit')\x1b[0m\n")
        return label

    def start(self):
        """Labeling interface that displays an image and prompts user for its class."""
        self.data_to_label = self.unlabeled_df[self.unlabeled_df.is_labeled == False]

        if self.num_imgs == 0:
            print("All data are labeled! \N{grinning face}")
            return
        try:
            for i in range(len(self.data_to_label)):
                # Clear display then show next image to label
                img_uid = self.data_to_label.iloc[i]['img_uid']
                self.plot_img(img_uid)

                display.clear_output(wait=True)
                plt.show()

                # Get image label
                time.sleep(0.002)  # Sleep to avoid issues with display clearing routine
                label = self.get_user_label()
                if label == 'exit':
                    break
                add_labeled_data(self.labeled_df, self.unlabeled_df, img_uid, self.user_uid, label)
                self.num_labeled += 1
                plt.close()
        except KeyboardInterrupt:
            return
        finally:
            display.clear_output(wait=True)
            print(self.get_progress_str())
            print('Exiting and saving your labels...')
            self.save_progress()
            print('Success!')

    def save_progress(self):
        batch_label_dir = get_batch_label_dir(self.task, self.batch_id, self.root_labeled_data_dir)

        save_df(self.img_df,
                'img',
                self.user_uid,
                self.batch_id,
                self.task,
                True,
                batch_label_dir
                )

        save_df(self.labeled_df,
                'labeled-data',
                self.user_uid,
                self.batch_id,
                self.task,
                True,
                batch_label_dir
                )

        save_df(self.unlabeled_df,
                'unlabeled-data',
                self.user_uid,
                self.batch_id,
                self.task,
                True,
                batch_label_dir
                )

    def create_export_zipfile(self):
        data_export_dir = get_data_export_dir(self.task, self.batch_id, self.user_uid)
        os.makedirs(data_export_dir, exist_ok=True)

        save_df(self.img_df,
                'img',
                self.user_uid,
                self.batch_id,
                self.task,
                False,
                data_export_dir
                )

        save_df(self.labeled_df,
                'labeled-data',
                self.user_uid,
                self.batch_id,
                self.task,
                False,
                data_export_dir
                )

        save_df(self.unlabeled_df,
                'unlabeled-data',
                self.user_uid,
                self.batch_id,
                self.task,
                False,
                data_export_dir
                )

        # Save user info
        user_info = {
            'name': self.name,
            'user-uid': self.user_uid
        }
        user_info_path = data_export_dir + '/' + 'user_info.json'
        with open(user_info_path, 'w') as f:
            f.write(json.dumps(user_info))

        shutil.make_archive(data_export_dir, 'zip', data_export_dir)
        shutil.rmtree(data_export_dir)
