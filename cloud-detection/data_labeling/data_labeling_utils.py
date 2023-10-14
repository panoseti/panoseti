#! /usr/bin/env python3

import hashlib
import os
import json
import math
import time
from datetime import datetime, timedelta, tzinfo

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from PIL import Image
#from IPython import display


plt.figure(figsize=(15, 15))

data_labels_file = 'skycam_labels.json'
with open(data_labels_file, 'r') as f:
    labels = json.load(f)

# Skycam directory utils

valid_image_types = ['original', 'cropped', 'pfov']

def get_img_subdirs(skycam_dir):
    """Return dict of skycam image directories."""
    img_subdirs = {}
    for img_type in valid_image_types:
        img_subdirs[img_type] = f'{skycam_dir}/{img_type}'
    return img_subdirs


def get_img_path(original_fname, img_type, skycam_dir):
    assert img_type in valid_image_types, f"{img_type} is not supported"
    img_subdirs = get_img_subdirs(skycam_dir)
    if original_fname[-4:] != '.jpg':
        return None
    if img_type == 'original':
        return f"{img_subdirs['original']}/{original_fname}"
    elif img_type == 'cropped':
        return f"{img_subdirs['cropped']}/{original_fname[:-4]}_cropped.jpg"
    elif img_type == 'pfov':
        return f"{img_subdirs['pfov']}/{original_fname[:-4]}_pfov.jpg"
    else:
        return None


def get_skycam_dir(skycam_type, year, month, day):
    if skycam_type == 'SC':
        return f'SC_imgs_{year}-{month:0>2}-{day:0>2}'
    elif skycam_type == 'SC2':
        return f'SC2_imgs_{year}-{month:0>2}-{day:0>2}'


def init_preprocessing_dirs(skycam_dir):
    """Initialize pre-processing directories."""
    img_subdirs = get_img_subdirs(skycam_dir)
    is_data_preprocessed(skycam_dir)
    #for dir_name in [img_subdirs['cropped'], img_subdirs['pfov']]:
    for dir_name in img_subdirs.values():
        os.makedirs(dir_name, exist_ok=True)


def is_initialized(skycam_dir):
    img_subdirs = get_img_subdirs(skycam_dir)
    if os.path.exists(skycam_dir) and len(os.listdir()) > 0:
        is_initialized = False
        for path in os.listdir():
            if path in img_subdirs:
                is_initialized |= len(os.listdir()) > 0
            if os.path.isfile(path):
                is_initialized = False
        if is_initialized:
            raise FileExistsError(f"Expected directory {skycam_dir} to be uninitialized, but found the following files:\n\t"
                                    f"{os.walk(skycam_dir)}")


def is_data_downloaded(skycam_dir):
    """Checks if data is already downloaded."""
    img_subdirs = get_img_subdirs(skycam_dir)
    if os.path.exists(img_subdirs['original']) and len(os.listdir(img_subdirs['original'])) > 0:
        raise FileExistsError(f"Data already downloaded at {img_subdirs['original']}")
    is_initialized(skycam_dir)

def is_data_preprocessed(skycam_dir):
    """Checks if data is already processed."""
    img_subdirs = get_img_subdirs(skycam_dir)
    if os.path.exists(img_subdirs['cropped']) and len(os.listdir(img_subdirs['cropped'])) > 0:
        raise FileExistsError(f"Data in {skycam_dir} already processed")
    is_initialized(skycam_dir)


def get_img_time(skycam_fname):
    """Returns datetime object based on the image timestamp contained in skycam_fname."""
    if skycam_fname[-4:] != '.jpg':
        raise Warning('Expected a .jpg file')
    # Example: SC2_20230625190102 -> SC2, 20230625190102
    skycam_type, t = skycam_fname[:-4].split('_')
    # 20230625190102 -> 2023, 06, 25, 19, 01, 02
    time_fields = t[0:4], t[4:6], t[6:8], t[8:10], t[10:12], t[12:14]
    year, month, day, hour, minute, second = [int(tf) for tf in time_fields]

    timestamp = datetime(year, month, day, hour, minute, second)
    return timestamp



# Plotting routines
def get_img_uid_to_data(img_metadata_df, skycam_dir):
    """Returns a function that maps uids of original skycam images to the corresponding image data array."""
    img_subdirs = get_img_subdirs(skycam_dir)

    def img_uid_to_data(img_uid, img_type):
        original_fname = (img_metadata_df.loc[(img_metadata_df.img_uid == img_uid), 'fname']).iloc[0]

        if img_type not in img_subdirs:
            raise Warning(f"'{img_type}' is not a valid skycam image type.")
        fpath = get_img_path(original_fname, img_type, skycam_dir)
        img = np.asarray(Image.open(fpath))
        return img

    return img_uid_to_data


def get_plot_img(skycam_df, skycam_dir):
    """Given an image uid, display the corresponding image."""
    img_uid_to_data = get_img_uid_to_data(skycam_df, skycam_dir)

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


def show_classifications(labeled_data_df, skycam_df, skycam_dir):
    """Display all classified images, organized by class."""
    for key in labels.keys():
        make_img_grid(labeled_data_df, skycam_df, skycam_dir, int(key))


def make_img_grid(labeled_data_df, skycam_df, skycam_dir, label, cols=8, rows_per_plot=8):
    """Grid of all classified images labeled as the given label"""
    img_uid_to_data = get_img_uid_to_data(skycam_df, skycam_dir)
    data_with_given_label = labeled_data_df.loc[(labeled_data_df.label == label), 'img_uid']
    imgs = [img_uid_to_data(img_uid, 'cropped') for img_uid in data_with_given_label]
    if len(imgs) == 0:
        print(f'No images labeled as "{labels[str(label)]}"')
        return
    else:
        print(f'Images you classified as "{labels[str(label)]}":')
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


# Labeling interface functions

def get_valid_labels_str(valid_labels):
    """
    Params:
        valid_labels: dictionary of image classes and their numeric encoding.
    Returns: string describing all valid labels and their encodings.
    """
    s = ""
    for key, val in labels.items():
        s += f"{int(key) + 1}='{val}', "
    return s[:-2]


def get_progress_str(total_itrs, itr_num, num_symbols=50):
    """Generate status progress bar"""
    num_divisions = min(num_symbols, total_itrs)
    itrs_per_symbol = num_symbols / total_itrs
    prog = int(itr_num * itrs_per_symbol)
    progress_bar = '\u2593' * prog + '\u2591' * (num_symbols - prog)
    msg = f"|{progress_bar}|" + f"\t{itr_num} / {total_itrs}\n"
    return msg


def get_user_label(total_itrs, itr_num):
    """Prompts and returns the label a user assigns to a given image."""
    valid_label_str = get_valid_labels_str(labels)
    progress_str = get_progress_str(total_itrs, itr_num)
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
        if label.isnumeric() and str(int(label) - 1) in labels:
            valid_label = True
            label = int(label) - 1
        elif label == 'exit':
            return 'exit'
        else:
            first_itr = False
            print(f"\x1b[31mError:\t   '{label}' is not a valid label. "
                  "(To exit the session, type 'exit')\x1b[0m\n")
    return label


def get_label_session(labeled_df, unlabeled_df, skycam_df, skycam_dir):
    """Constructor for labeling interface.
    Params:
        df: DataFrame containing metadata about unlabeled data points.
        labeler_name: Name of person doing the labeling.
    """
    data_to_label = unlabeled_df[unlabeled_df.is_labeled == False]
    # print(data_to_label)
    num_imgs = len(data_to_label)
    plot_img = get_plot_img(skycam_df, skycam_dir)

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
                label = get_user_label(num_imgs, i)
                if label == 'exit':
                    break
                add_labeled_data(labeled_df, unlabeled_df, img_uid, user_uid, label)
                num_labeled += 1
                plt.close()
        except KeyboardInterrupt:
            return
        finally:
            display.clear_output(wait=True)
            print(get_progress_str(num_imgs, num_labeled))
            print('Exiting and saving your labels...')

    return label_session