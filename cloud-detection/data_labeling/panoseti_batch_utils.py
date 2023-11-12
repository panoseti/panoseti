
import json
import os
import numpy as np
import seaborn_image as isns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors


pano_path_index_fname = 'pano_path_index.json'
valid_image_types = ['original', 'derivative', 'fft']

# File structure abstraction
def get_pano_subdirs(pano_path):
    pano_subdirs = {}
    for img_type in valid_image_types:
        pano_subdirs[img_type] = f'{pano_path}/{img_type}'
    return pano_subdirs

def get_skycam_img_path(original_fname, img_type, run_dir):
    assert img_type in valid_image_types, f"{img_type} is not supported"
    pano_subdirs = get_pano_subdirs(run_dir)
    if original_fname[-4:] != '.jpg':
        return None
    if img_type == 'original':
        return f"{pano_subdirs['original']}/{original_fname}"
    elif img_type == 'derivative':
        return f"{pano_subdirs['derivative']}/{original_fname[:-4]}_cropped.jpg"
    elif img_type == 'fft':
        return f"{pano_subdirs['fft']}/{original_fname[:-4]}_pfov.jpg"
    else:
        return None

# Plotting
def plot_image_grid(imgs, delta_ts, nr, vmin=-4, vmax=4):
    for i in range(len(imgs)):
        if imgs[i] is None or not isinstance(imgs[i], np.ndarray):
            print('no image')
            return None
        if imgs[i].shape != (32, 32):
            imgs[i] = np.reshape(imgs[i], (32, 32))
    plt.rcParams.update({'axes.titlesize': 'small'})
    fig = plt.figure(figsize=(2.5, 4.0))

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nr, 1),  # creates nr x 1 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     share_all=True
                     )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    titles = []
    for dt in delta_ts:
        titles.append(f'{dt} s')
    fig.suptitle(f': derivative', ha='center')

    ims = []
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for i, (ax, img, title) in enumerate(zip(grid, imgs, titles)):
        im = ax.imshow(img, aspect='equal', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title, x=-0.3, y=0.35)
        im.set_norm(norm)
        ims.append(im)
    fig.colorbar(ims[0], ax=grid)
    return fig

def plot_image_grid_isns(imgs, delta_ts):
    for i in range(len(imgs)):
        if imgs[i] is None or not isinstance(imgs[i], np.ndarray):
            print('no image')
            return None
        if imgs[i].shape != (32, 32):
            imgs[i] = np.reshape(imgs[i], (32, 32))
    # print(delta_ts)
    # ax = isns.ImageGrid(imgs, height=1.5, col_wrap=1, vmin=-100, vmax=100, cmap="viridis", cbar=False)
    ax = isns.ImageGrid(imgs, height=1.5, col_wrap=1, vmin=-3, vmax=3, cmap="viridis", cbar=False)
    return ax.fig

def plot_image_fft(img):
    if img is None or not isinstance(img, np.ndarray):
        print('no image')
        return None
    if img.shape != (32, 32):
        img = np.reshape(img, (32, 32))
    ax = isns.fftplot(img, cmap="viridis", window_type='cosine')
    return ax.get_figure()
