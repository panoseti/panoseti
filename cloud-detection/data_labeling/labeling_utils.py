import hashlib
import os
import json
import math
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from PIL import Image
from IPython import display

from skycam_utils import get_img_subdirs, get_img_path, get_skycam_dir

plt.figure(figsize=(15, 15))

class LabelSession:
    data_labels_file = 'skycam_labels.json'
    data_batch_dir = 'data_batches'

    def __init__(self, name, batch_id=0):
        self.name = name
        self.user_uid = get_uid(name)

        self.batch_id = batch_id
        self.batch_dir = get_batch_dir(LabelSession.data_batch_dir, batch_id)

        self.img_df = get_dataframe('img')
        self.unlabeled_df = get_dataframe('unlabeled_data')
        self.labeled_df = get_dataframe('labeled_data')

        with open(LabelSession.data_labels_file, 'r') as f:
            self.labels = json.load(f)

    # Plotting routines
    def get_img_uid_to_data(self, img_df, skycam_dir):
        """Returns a function that maps uids of original skycam images to the corresponding image data array."""
        img_subdirs = get_img_subdirs(skycam_dir)

        def img_uid_to_data(img_uid, img_type):
            original_fname = (self.img_df.loc[(self.img_df.img_uid == img_uid), 'fname']).iloc[0]

            if img_type not in img_subdirs:
                raise Warning(f"'{img_type}' is not a valid skycam image type.")
            fpath = get_img_path(original_fname, img_type, skycam_dir)
            img = np.asarray(Image.open(fpath))
            return img

        return img_uid_to_data


    def get_plot_img(self, skycam_df, skycam_dir):
        """Given an image uid, display the corresponding image."""
        img_uid_to_data = self.get_img_uid_to_data(skycam_df, skycam_dir)

        def plot_img(img_uid):
            fig = plt.figure(figsize=(15, 15))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                             nrows_ncols=(1, 2),  # creates width x height grid of axes
                             axes_pad=0.3,  # pad between axes in inch.
                             share_all=False,
                             )
            imgs = [img_uid_to_data(img_uid, 'cropped'),
                    img_uid_to_data(img_uid, 'pfov')]
            titles = [f'{img_uid[:8]}: cropped fov',
                      f'{img_uid[:8]}: full img with pfov']

            for i, (ax, img, title) in enumerate(zip(grid, imgs, titles)):
                ax.imshow(img, aspect='equal')
                ax.set_title(title)
            return fig

        return plot_img

    def show_classifications(self, labeled_data_df, skycam_df, skycam_dir):
        """Display all labeled images, organized by assigned class."""
        for key in self.labels.keys():
            self.make_img_grid(labeled_data_df, skycam_df, skycam_dir, int(key))


    def make_img_grid(self, labeled_data_df, skycam_df, skycam_dir, label, cols=8, rows_per_plot=8):
        """Grid of all classified images labeled as the given label"""
        img_uid_to_data = self.get_img_uid_to_data(skycam_df, skycam_dir)
        data_with_given_label = labeled_data_df.loc[(labeled_data_df.label == label), 'img_uid']
        imgs = [img_uid_to_data(img_uid, 'cropped') for img_uid in data_with_given_label]
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

    def get_valid_labels_str(self, valid_labels):
        """
        Params:
            valid_labels: dictionary of image classes and their numeric encoding.
        Returns: string describing all valid labels and their encodings.
        """
        s = ""
        for key, val in self.labels.items():
            s += f"{int(key) + 1}='{val}', "
        return s[:-2]


    def get_progress_str(self, total_itrs, itr_num, num_symbols=50):
        """Generate status progress bar"""
        num_divisions = min(num_symbols, total_itrs)
        itrs_per_symbol = num_symbols / total_itrs
        prog = int(itr_num * itrs_per_symbol)
        progress_bar = '\u2593' * prog + '\u2591' * (num_symbols - prog)    # red text
        msg = f"|{progress_bar}|" + f"\t{itr_num} / {total_itrs}\n"
        return msg


    def get_user_label(self, total_itrs, itr_num):
        """Prompts and returns the label a user assigns to a given image."""
        valid_label_str = self.get_valid_labels_str(self.labels)
        progress_str = self.get_progress_str(total_itrs, itr_num)
        valid_label = False
        first_itr = True
        while not valid_label:
            if first_itr:
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

    def get_label_session(self, labeled_df, unlabeled_df, skycam_df, skycam_dir):
        """Constructor for labeling interface.
        Params:
            df: DataFrame containing metadata about unlabeled data points.
            labeler_name: Name of person doing the labeling.
        """
        data_to_label = unlabeled_df[unlabeled_df.is_labeled == False]
        # print(data_to_label)
        num_imgs = len(data_to_label)
        plot_img = self.get_plot_img(skycam_df, skycam_dir)

        def label_session(user_uid):
            """Labeling interface that displays an image and prompts user for its class."""
            if num_imgs == 0:
                print("All data are labeled! \N{grinning face}")
                return
            num_labeled = 0
            try:
                for i in range(len(data_to_label)):
                    # Clear display then show next image to label
                    img_uid = data_to_label.iloc[i]['img_uid']
                    plot_img(img_uid)

                    display.clear_output(wait=True)
                    plt.show()

                    # Get image label
                    time.sleep(0.001)  # Sleep to avoid issues with display clearing routine
                    label = self.get_user_label(num_imgs, i)
                    if label == 'exit':
                        break
                    add_labeled_data(labeled_df, unlabeled_df, img_uid, user_uid, label)
                    num_labeled += 1
                    plt.close()
            except KeyboardInterrupt:
                return
            finally:
                display.clear_output(wait=True)
                print(self.get_progress_str(num_imgs, num_labeled))
                print('Exiting and saving your labels...')

        return label_session



# Database formats
def get_dataframe_formats():
    dataframe_formats = {
        'user': ['user_uid', 'name'],
        'img': ['img_uid', 'fname', 'skycam', 'timestamp'],
        'unlabeled_data': ['img_uid', 'is_labeled'],
        'labeled_data': ['img_uid', 'user_uid', 'label']
    }
    return dataframe_formats

def get_dataframe(df_type):
    dataframe_formats = get_dataframe_formats()
    assert df_type in dataframe_formats, f"'{df_type}' is not a supported dataframe format"
    return pd.DataFrame(columns=dataframe_formats[df_type])

def get_df_save_name(user_uid, batch_id, df_type):
    # TODO: reverse order
    return "user-uid_{0}.batch-id_{1}.type_{2}.csv".format(user_uid, batch_id, df_type)

def save_df(user_uid, batch_id, df_type):
    dataframe_formats = get_dataframe_formats()
    for df_type, df_format in dataframe_formats.items():
        df = get_dataframe(df_type)
        df.to_csv(get_df_save_name())

def load_df(batch_id, df_type):
    ...

# Database IO routines



def load_database_formats(data_batch_dir, batch_id):
    ...

def get_batch_dir(data_batch_dir, batch_id):
    return "task_cloud-detection.batch_{0}".format(batch_id)


# Database interface routines

def get_uid(data: str):
    """Returns SHA1 hash of a string input data."""
    data_bytes = data.encode('utf-8')
    data_uid = hashlib.sha1(data_bytes).hexdigest()
    return data_uid


def add_user(user_df, name, verbose=False):
    """Adds new user to user_df."""
    user_uid = get_uid(name)
    if not user_df.loc[:, 'user_uid'].str.contains(user_uid).any():
        user_df.loc[len(user_df)] = [user_uid, name]
    elif verbose:
        print(f'An entry for "{name}" already exists')
    return user_uid


def add_skycam_img(skycam_df, fname, skycam, verbose=False):
    img_uid = get_uid(fname)
    if not skycam_df.loc[:, 'img_uid'].str.contains(img_uid).any():
        skycam_df.loc[len(skycam_df)] = [img_uid, fname, skycam]
    elif verbose:
        print(f'An entry for "{fname}" already exists')
    return img_uid


def add_unlabeled_data(unlabeled_df, img_uid, verbose=False):
    if not unlabeled_df.loc[:, 'img_uid'].str.contains(img_uid).any():
        unlabeled_df.loc[len(unlabeled_df)] = [img_uid, False]
    elif verbose:
        print(f'An entry for "{img_uid}" already exists')


def add_labeled_data(labeled_df, unlabeled_df, img_uid, user_uid, label):
    # labeled_df.loc[(labeled_df['img_uid'] == img_uid), ['user_uid', 'label']] = [user_uid, label]
    labeled_df.loc[len(labeled_df)] = [img_uid, user_uid, label]
    unlabeled_df.loc[(unlabeled_df['img_uid'] == img_uid), 'is_labeled'] = True


def init_skycam_df(skycam_df, unlabeled_data_df, skycam_dir):
    img_subdirs = get_img_subdirs(skycam_dir)
    for fname in os.listdir(img_subdirs['original']):
        if fname[-4:] == '.jpg':
            add_skycam_img(skycam_df, fname, 'SC2')
            add_unlabeled_data(unlabeled_data_df, get_uid(fname))

