import os
import sys
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

sys.path.append('../')
from skycam_utils import get_skycam_img_path, get_batch_dir, skycam_path_index_fname, get_skycam_root_path
from panoseti_batch_utils import get_pano_img_path, get_pano_subdirs, get_pano_root_path
from dataframe_utils import *

class LabelSession:
    data_labels_path = f'../{data_labels_fname}'
    root_batch_labels_dir = './batch_labels'

    def __init__(self, name, batch_id, task='cloud-detection'):
        self.name = name
        if name == "YOUR NAME":
            raise ValueError(f"Please enter your full name")
        self.user_uid = get_uid(name)
        self.batch_id = batch_id
        self.task = task

        self.batch_dir_name = get_batch_dir(task, batch_id)
        self.batch_data_path = f'{batch_data_root_dir}/{self.batch_dir_name}'
        self.batch_labels_path = f'{self.root_batch_labels_dir}/{self.batch_dir_name}'
        self.skycam_imgs_path = get_skycam_root_path(self.batch_data_path)
        self.pano_imgs_path = get_pano_root_path(self.batch_data_path)

        # Unzip batched data, if it exists
        os.makedirs(batch_data_root_dir, exist_ok=True)
        os.makedirs(self.batch_labels_path, exist_ok=True)
        try:
            unpack_batch_data(batch_data_root_dir)
            with open(f'{self.batch_data_path}/{skycam_path_index_fname}', 'r') as f:
                self.skycam_paths = json.load(f)
            with open(f'{self.batch_data_path}/{pano_path_index_fname}', 'r') as f:
                self.pano_paths = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find \x1b[31m{self.batch_dir_name}\x1b[0m\n"
                                    f"Try adding the zipped data batch file to the following directory:\n"
                                    f"\x1b[31m{os.path.abspath(batch_data_root_dir)}\x1b[0m")
        try:
            with open(self.data_labels_path, 'r') as f:
                self.labels = json.load(f)
        except:
            raise FileNotFoundError(f"Could not find the file {self.data_labels_path}\n"
                                    f"Try pulling the file from the panoseti github.")
        # Init dataframes
        self.loaded_dfs_from_file = {}
        self.feature_df = self.init_dataframe('feature')
        self.skycam_df = self.init_dataframe('skycam')
        self.pano_df = self.init_dataframe('pano')
        self.unlabeled_df = self.init_dataframe('unlabeled')
        self.labeled_df = self.init_dataframe('labeled')
        self.data_to_label = None

        self.num_imgs = len(self.unlabeled_df)
        self.num_labeled = len(self.unlabeled_df.loc[self.unlabeled_df.is_labeled == True])

    def init_dataframe(self, df_type):
        """Attempt to load the given dataframe from file, if it exists. Otherwise, create a new dataframe."""
        if df_type in ['feature', 'skycam', 'pano']:
            # These dataframes are initialized by make_batch.py and should exist within the downloaded batch data.
            df = load_df(
                None, self.batch_id, df_type, self.task, is_temp=False,
                save_dir=self.batch_data_path
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
                    df = df.sample(frac=1).reset_index(drop=True)
        else:
            raise ValueError(f'Unsupported df_type: "{df_type}"')
        return df

    def skycam_uid_to_data(self, skycam_uid, img_type):
        original_fname, skycam_dir = self.skycam_df.loc[
            (self.skycam_df.skycam_uid == skycam_uid), ['fname', 'skycam_dir']
        ].iloc[0]
        fpath = get_skycam_img_path(original_fname, img_type, f'{self.skycam_imgs_path}/{skycam_dir}')
        img = np.asarray(Image.open(fpath))
        return img

    def pano_uid_to_data(self, pano_uid, img_type):
        run_dir = self.pano_df.loc[
            (self.pano_df.pano_uid == pano_uid), 'run_dir'
        ].iloc[0]
        fpath = get_pano_img_path(f'{self.pano_imgs_path}/{run_dir}', pano_uid, img_type)
        img = np.asarray(Image.open(fpath))
        return img

    def add_subplot(self, ax, img, title, aspect=None):
        ax.imshow(img, aspect=aspect)
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_title(title)

    def plot_img(self, feature_uid):
        skycam_uid, pano_uid = self.feature_df.loc[
            self.feature_df.feature_uid == feature_uid, ['skycam_uid', 'pano_uid']
        ].iloc[0]

        # creating grid for subplots
        fig = plt.figure()
        fig.set_figheight(6)
        fig.set_figwidth(17)

        shape = (8, 22)
        ax0 = plt.subplot2grid(shape=shape, loc=(0, 4), colspan=10, rowspan=8)

        # ax1 = plt.subplot2grid(shape=shape, loc=(0, 12), colspan=2, rowspan=2)
        ax2 = plt.subplot2grid(shape=shape, loc=(0, 0), colspan=4, rowspan=4)
        ax3 = plt.subplot2grid(shape=shape, loc=(4, 0), colspan=4, rowspan=4)

        ax4 = plt.subplot2grid(shape=shape, loc=(0, 14), colspan=8, rowspan=4)
        ax5 = plt.subplot2grid(shape=shape, loc=(4, 14), colspan=8, rowspan=4)



        # plotting subplots
        self.add_subplot(
            ax0,
            self.skycam_uid_to_data(skycam_uid, 'pfov'),
            f'{skycam_uid[:8]}: skycam w/ panofov'
        )
        # self.add_subplot(
        #     ax1,
        #     self.skycam_uid_to_data(skycam_uid, 'cropped'),
        #     f'{skycam_uid[:8]}: skycam pfov'
        # )
        self.add_subplot(
            ax2,
            self.pano_uid_to_data(pano_uid, 'original'),
            f'{pano_uid[:8]}: pano normed'
        )
        self.add_subplot(
            ax3,
            self.pano_uid_to_data(pano_uid, 'fft'),
            f'{pano_uid[:8]}: pano fft'
        )
        self.add_subplot(
            ax4,
            self.pano_uid_to_data(pano_uid, 'derivative'),
            f'{pano_uid[:8]}: time derivatives'
        )
        self.add_subplot(
            ax5,
            self.pano_uid_to_data(pano_uid, 'fft-derivative'),
            f'{pano_uid[:8]}: time derivative ffts'
        )


        # automatically adjust padding horizontally
        # as well as vertically.
        plt.tight_layout()
        return fig


    # Plotting routines
    def plot_img_old(self, feature_uid):
        """Given an image uid, display the corresponding image features."""
        fig = plt.figure(figsize=(24, 24))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(3, 1),  # creates width x height grid of axes
                         axes_pad=0.3,  # pad between axes in inch.
                         share_all=False,
                         )
        skycam_uid, pano_uid = self.feature_df.loc[
            self.feature_df.feature_uid == feature_uid, ['skycam_uid', 'pano_uid']
        ].iloc[0]

        imgs = [
            self.pano_uid_to_data(pano_uid, 'fft-derivative'),
            self.pano_uid_to_data(pano_uid, 'derivative'),
            self.skycam_uid_to_data(skycam_uid, 'pfov'),
            #self.skycam_uid_to_data(skycam_uid, 'cropped'),
        ]
        titles = [
            f'{pano_uid[:8]}: time derivative ffts',
            f'{pano_uid[:8]}: time derivatives',
            f'{skycam_uid[:8]}: full img with pfov'
            #f'{skycam_uid[:8]}: cropped fov',
        ]

        for i, (ax, img, title) in enumerate(zip(grid, imgs, titles)):
            print(i)
            ax.imshow(img)#, aspect='equal')
            ax.set_title(title)
        return fig

    def show_classifications(self):
        """Display all labeled images, organized by assigned class."""
        for key in self.labels.keys():
            self.make_img_grid(self.labels[key])


    def make_img_grid(self, label, cols=8, rows_per_plot=8):
        """Grid of all classified images labeled as the given label"""
        feature_uids_with_given_label = self.labeled_df.loc[
            (self.labeled_df.label == label), 'feature_uid'
        ]
        # skycam_uids_with_given_label = self.feature_df.loc[
        #     (self.feature_df['feature_uid'].isin(feature_uids_with_given_label)), 'skycam_uid'
        # ]
        pano_uids_with_given_label = self.feature_df.loc[
            (self.feature_df['feature_uid'].isin(feature_uids_with_given_label)), 'pano_uid'
        ]
        imgs = []
        for pano_uid in pano_uids_with_given_label:
            imgs.append(self.pano_uid_to_data(pano_uid, 'original'))
        if len(imgs) == 0:
            print(f'No images labeled as "{label}"')
            return
        else:
            print(f'Images you classified as "{label}":')
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
                    feature_uid = feature_uids_with_given_label.iloc[img_idx]
                    ax.set_title(f'{23}{feature_uid[:6]}')  # Label each plot with first 6 chars of feature_uid
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

        if len(self.data_to_label) == 0:
            emojis = ['🌈', '💯', '✨', '🎉', '🎃', '🔭', '🌌']
            print(f"All data are labeled! {np.random.choice(emojis)}")
            return
        try:
            for i in range(len(self.data_to_label)):
                # Clear display then show next image to label
                feature_uid = self.data_to_label.iloc[i]['feature_uid']
                self.plot_img(feature_uid)

                display.clear_output(wait=True)
                plt.show()

                # Get image label
                time.sleep(0.002)  # Sleep to avoid issues with display clearing routine
                label_idx = self.get_user_label()
                if label_idx == 'exit':
                    break
                label_str = self.labels[str(label_idx)]
                self.labeled_df = add_labeled_data(self.labeled_df, self.unlabeled_df, feature_uid, self.user_uid, label_str)
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
        self.data_to_label = self.unlabeled_df[self.unlabeled_df.is_labeled == False]
        if not allow_partial_batch and len(self.data_to_label) > 0:
            print(f'Please label all data in the batch before exporting.')
            return
        print('Zipping batch labels...')
        data_export_dir = get_data_export_dir(self.task, self.batch_id, self.user_uid, root='.')
        os.makedirs(data_export_dir, exist_ok=True)

        save_df(self.labeled_df,
                'labeled',
                self.user_uid,
                self.batch_id,
                self.task,
                False,
                data_export_dir
                )

        save_df(self.unlabeled_df,
                'unlabeled',
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
        print('Done!')


if __name__ == '__main__':
    session = LabelSession('Nico', 0)
    session.start()
