"""
Utility functions for building panoseti features and batch data directories
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import colors
from scipy.fftpack import fftn, fftshift
from skimage.filters import window
import seaborn_image as isns
import seaborn as sns

from dataframe_utils import add_pano_img
from batch_building_utils import *



# File structure abstraction
def make_pano_paths_json(batch_path):
    """Create file for indexing sky-camera image paths."""
    assert os.path.exists(batch_path), f"Could not find the batch directory {batch_path}"
    pano_paths = {}
    pano_imgs_root_path = get_pano_root_path(batch_path)
    for path in os.listdir(pano_imgs_root_path):
        pano_path = f'{pano_imgs_root_path}/{path}'
        if os.path.isdir(pano_path) and 'pffd' in path:
            pano_paths[pano_path] = {
                "img_subdirs": {},
                "imgs_per_subdir": -1,
            }
            pano_subdirs = get_pano_subdirs(pano_path)
            pano_paths[pano_path]["img_subdirs"] = pano_subdirs
            num_imgs_per_subdir = []
            for subdir in pano_subdirs.values():
                num_imgs_per_subdir.append(len(os.listdir(subdir)))
            if not all([e == num_imgs_per_subdir[0] for e in num_imgs_per_subdir]):
                raise Warning(f"Unequal number of images in {pano_path}")
            pano_paths[pano_path]["imgs_per_subdir"] = num_imgs_per_subdir[0]
    with open(f"{batch_path}/{pano_path_index_fname}", 'w') as f:
        f.write(json.dumps(pano_paths, indent=4))
    return pano_paths



def add_pano_data_to_pano_df(pano_df, batch_id, pano_imgs_root_path, pano_dir, verbose):
    """Add entries for each skycam image to skycam_df """
    original_img_dir = get_pano_subdirs(f'{pano_imgs_root_path}/{pano_dir}')['original']
    for original_fname in os.listdir(original_img_dir):
        if original_fname[-4:] == '.jpg':
            # Collect image features

            # Add entries to skycam_df
            pano_df = add_pano_img(pano_df, ...)
    return pano_df


# Plotting
def plot_fft_time_derivative(imgs, delta_ts, nc, vmin, vmax, cmap):
    for i in range(len(imgs)):
        if imgs[i] is None or not isinstance(imgs[i], np.ndarray):
            print('no image')
            return None
        if imgs[i].shape != (32, 32):
            imgs[i] = np.reshape(imgs[i], (32, 32))
    # plt.rcParams.update({'axes.titlesize': 'small'})
    fig = plt.figure(figsize=(10., 3.))

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, nc),  # creates nr x 1 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     share_all=True
                     )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    titles = []
    for dt in delta_ts:
        titles.append(f'{dt} s')
    # fig.suptitle(f': derivative', ha='center')

    ims = []
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for i, (ax, img, title) in enumerate(zip(grid, imgs, titles)):
        #im = ax.imshow(img, aspect='equal', cmap='viridis', vmin=vmin, vmax=vmax)
        # im = isns.fftplot(img, aspect='equal', cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)
        im = plot_image_fft(img, ax=ax, cmap=cmap)
        ax.set_title(title, x=0.5, y=-0.2)
        #im.set_norm(norm)
        ims.append(im)
    return fig

def plot_time_derivative(imgs, delta_ts, vmin, vmax, cmap):
    for i in range(len(imgs)):
        if imgs[i] is None or not isinstance(imgs[i], np.ndarray):
            print('no image')
            return None
        if imgs[i].shape != (32, 32):
            imgs[i] = np.reshape(imgs[i], (32, 32))
    # print(delta_ts)
    # ax = isns.ImageGrid(imgs, height=1.5, col_wrap=1, vmin=-100, vmax=100, cmap="viridis", cbar=False)
    titles = []
    for dt in delta_ts:
        titles.append(f'{dt} s')
    #print('len', len(imgs))
    ax = isns.ImageGrid(imgs,
                        height=3,
                        aspect=0.75,
                        col_wrap=4,
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap,
                        cbar_label=titles,
                        orientation="h")
    fig = ax.fig
    # fig.suptitle(f': derivative', ha='center')
    return fig

def apply_fft(data):
    if data is None or not isinstance(data, np.ndarray):
        print('no image')
        return None
    if data.shape != (32, 32):
        data = np.reshape(data, (32, 32))

    shape = data.shape
    window_type = "cosine"
    data = data * window(window_type, shape)
    data = np.abs(fftn(data))
    data = fftshift(data)
    data = np.log(data)
    return data


def plot_image_fft(data, cmap, **kwargs):
    if data is None or not isinstance(data, np.ndarray):
        print('no image')
        return None
    if data.shape != (32, 32):
        data = np.reshape(data, (32, 32))

    fft_data = apply_fft(data)

    ax = isns.imgplot(
        fft_data,
        cmap=cmap,
        cbar=False,
        showticks=False,
        describe=False,
        despine=None,
        **kwargs
    )
    return ax.get_figure()




if __name__ == '__main__':
    make_pano_paths_json('/panoseti-software/cloud-detection/data_labeling/batch_data1/task_cloud-detection.batch-id_0')
