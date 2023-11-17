"""
PanosetiBatchBuilder creates all panoseti features for a single observing run.
"""

import os
import numpy as np
import sys
from collections import deque
from datetime import timedelta

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn_image as isns
import seaborn as sns
from matplotlib import colors
from scipy.fftpack import fftn, fftshift
from skimage.filters import window

from panoseti_file_interfaces import ObservingRunFileInterface
from skycam_utils import get_batch_dir, get_skycam_img_time, get_unix_from_datetime, get_skycam_root_path, get_skycam_subdirs
from panoseti_batch_utils import *
from dataframe_utils import *

sys.path.append("../../util")
import pff


class PanosetiBatchBuilder(ObservingRunFileInterface):

    def __init__(self, data_dir, run_dir, task, batch_id, force_recreate=False):
        super().__init__(data_dir, run_dir)
        self.task = task
        self.batch_id = batch_id
        self.batch_dir = get_batch_dir(task, batch_id)
        self.batch_path = f'{batch_data_root_dir}/{self.batch_dir}'
        self.pano_path = f'{self.batch_path}/{pano_imgs_root_dir}/{run_dir}'
        self.pano_subdirs = get_pano_subdirs(self.pano_path)
        self.force_recreate = force_recreate
        os.makedirs(self.pano_path, exist_ok=True)

    def init_preprocessing_dirs(self):
        """Initialize pre-processing directories."""
        try:
            for dir_name in self.pano_subdirs.values():
                os.makedirs(dir_name, exist_ok=True)
            self.is_data_preprocessed()
        except FileExistsError as ferr:
            raise ferr

    def is_initialized(self):
        """Are pano subdirs initialized?"""
        if os.path.exists(self.pano_path) and len(os.listdir()) > 0:
            is_initialized = False
            for path in os.listdir():
                if path in self.pano_subdirs:
                    is_initialized |= len(os.listdir()) > 0
                if os.path.isfile(path):
                    is_initialized = False
            if is_initialized:
                raise FileExistsError(
                    f"Expected directory {self.pano_path} to be uninitialized, but found the following files:\n\t"
                    f"{os.walk(self.pano_path)}")

    def is_data_preprocessed(self):
        """Checks if data is already processed."""
        if os.path.exists(f'{self.batch_path}/{pano_path_index_fname}'):
            raise FileExistsError(f"Data in {self.pano_path} already processed")
        self.is_initialized()

    def iterate_module_files(self, module_id, step_size, verbose=False):
        """On a sample of the frames in the file represented by file_info, add the total
        image brightness to the data array beginning at data_offset."""
        module_pff_files = self.obs_pff_files[module_id]
        frame_offset = 0  # For roughly even frame step size across file boundaries
        for i in range(len(module_pff_files)):
            file_info = module_pff_files[i]
            fpath = f"{self.run_path}/{file_info['fname']}"
            if verbose: print(f"Processing {file_info['fname']}")
            with open(fpath, 'rb') as fp:
                # Start file pointer with an offset based on the previous file -> ensures even frame sampling
                fp.seek(
                    frame_offset * self.frame_size,
                    os.SEEK_CUR
                )
                new_nframes = file_info['nframes'] - frame_offset
                for _ in range(new_nframes // step_size):
                    j, img = self.read_frame(fp, step_size)
                    # TODO: do something here
                frame_offset = file_info['nframes'] - (new_nframes // step_size) * step_size

    def module_file_time_seek(self, module_id, target_time):
        """Search module data to find the frame with timestamp closest to target_time.
        target_time should be a unix timestamp."""
        module_pff_files = self.obs_pff_files[module_id]
        # Use binary search to find the file that contains target_time
        l = 0
        r = len(module_pff_files) - 1
        m = -1
        while l <= r:
            m = (r + l) // 2
            file_info = module_pff_files[m]
            if target_time > file_info['last_unix_t']:
                l = m + 1
            elif target_time < file_info['first_unix_t']:
                r = m - 1
            else:
                break
        file_info = module_pff_files[m]
        if target_time < file_info['first_unix_t'] or target_time > file_info['last_unix_t']:
            return None
        # Use binary search to find the frame closest to target_time
        fpath = f"{self.run_path}/{file_info['fname']}"
        with open(fpath, 'rb') as fp:
            frame_time = self.intgrn_usec * 10 ** (-6)
            pff.time_seek(fp, frame_time, self.img_size, target_time)
            frame_offset = int(fp.tell() / self.frame_size)
            j, img = self.read_frame(fp, self.img_bpp)
            frame_unix_t = pff.img_header_time(j)
            ret = {
                'file_idx': m,
                'frame_offset': frame_offset,
                'frame_unix_t': frame_unix_t
            }
            # j, img = self.read_frame(fp, self.img_bpp)
            # fig = self.plot_image(img)
            # plt.show()
            # plt.pause(2)
            # plt.close(fig)
            return ret

    def img_transform(self, img):
        img = np.reshape(img, (32, 32))
        # img = np.flip(img, axis=0)
        img = np.rot90(img, 2)
        return img

    def make_original_fig(self, start_file_idx, start_frame_offset, module_id, vmin, vmax, cmap, verbose=False):
        module_pff_files = self.obs_pff_files[module_id]
        file_info = module_pff_files[start_file_idx]
        fpath = f"{self.run_path}/{file_info['fname']}"
        with open(fpath, 'rb') as fp:
            fp.seek(
                start_frame_offset * self.frame_size,
                os.SEEK_CUR
            )
            j, img = self.read_frame(fp, self.img_bpp)
            img = self.img_transform(img)
            #img = (img - np.median(img)) / np.std(img)
            fig = self.plot_image(img, vmin=vmin, vmax=vmax, bins=40, cmap=cmap, perc=(0.5, 99.5))
            #plt.pause(0.5)
            return fig

    def make_fft_fig(self, start_file_idx, start_frame_offset, module_id, vmin, vmax, cmap, verbose=False):
        module_pff_files = self.obs_pff_files[module_id]
        file_info = module_pff_files[start_file_idx]
        fpath = f"{self.run_path}/{file_info['fname']}"
        with open(fpath, 'rb') as fp:
            fp.seek(
                start_frame_offset * self.frame_size,
                os.SEEK_CUR
            )
            j, img = self.read_frame(fp, self.img_bpp)
            img = self.img_transform(img)
            fig = plot_image_fft(img, vmin=vmin, vmax=vmax, cmap=cmap)
            return fig

    def make_time_derivative_figs(self,
                                  start_file_idx,
                                  start_frame_offset,
                                  module_id,
                                  step_delta_t,
                                  max_delta_t,
                                  ncols,
                                  vmin,
                                  vmax,
                                  cmap,
                                  verbose=False):
        """Compute time derivative feature relative to the frame specified by start_file_idx and start_frame_offset.
        Returns None if time derivative calc is not possible.

        Parameters
        @param start_file_idx: file containing reference frame
        @param start_frame_offset: number of frames to the reference frame
        @param module_id: module id number, as computed from its ip address
        @param step_delta_t: time step between sampled frames
        @param max_delta_t: max time step for derivative calculation
        @param ncols: number of evenly spaced time-derivatives
        """
        module_pff_files = self.obs_pff_files[module_id]

        frame_step_size = int(step_delta_t / (self.intgrn_usec * 1e-6))
        assert frame_step_size > 0
        hist_size = int(max_delta_t / step_delta_t)

        # Check if it is possible to construct a time-derivative with the given parameters and data
        with open(f"{self.run_path}/{module_pff_files[start_file_idx]['fname']}", 'rb') as f:
            f.seek(start_frame_offset * self.frame_size)
            j, img = self.read_frame(f, self.img_bpp)
            curr_unix_t = pff.img_header_time(j)
            s = curr_unix_t
            if (curr_unix_t - max_delta_t) < module_pff_files[0]['first_unix_t']:
                return None, None

        frame_offset = start_frame_offset
        hist = list()
        # Iterate backwards through the files until hist_size frames have been accumulated
        for i in range(start_file_idx, -1, -1):
            if len(hist) == hist_size:
                break
            file_info = module_pff_files[i]
            fpath = f"{self.run_path}/{file_info['fname']}"
            if verbose: print(f"Processing {file_info['fname']}")
            with open(fpath, 'rb') as fp:
                if verbose: print("newfile")
                # Start file pointer with an offset based on the previous file -> ensures even frame sampling
                fp.seek(
                    frame_offset * self.frame_size,
                    os.SEEK_CUR
                )
                # Iterate backwards through the file
                for j, img in self.frame_iterator(fp, (-1 * frame_step_size) + 1):
                    if j is None or img is None:
                        break
                    if len(hist) < hist_size:
                        hist.insert(0, img)
                        # if verbose: print(int(pff.img_header_time(j) - s))
                        continue
                    imgs = list()
                    delta_ts = []
                    for k in [int(i * hist_size / ncols) for i in range(ncols, 0, -1)]:
                        delta_ts.append(str(-step_delta_t * k))
                        data = (img - hist[k - 1]) / np.std(hist)
                        data = self.img_transform(data)
                        imgs.append(data)
                    # print(delta_ts)

                    fig_time_derivative = plot_time_derivative(
                        imgs, delta_ts, vmin=vmin[0], vmax=vmax[0], cmap=cmap[0]
                    )
                    fig_fft_time_derivative = plot_fft_time_derivative(
                        imgs, delta_ts, ncols, vmin[1], vmax[1], cmap=cmap[1]
                    )

                    # plt.pause(0.5)
                    return fig_time_derivative, fig_fft_time_derivative
                # Compute the frame offset for the next pff file
                if i > 0:
                    next_file_size = module_pff_files[i-1]['nframes'] * self.frame_size
                    curr_byte_offset = frame_step_size * self.frame_size - fp.tell()
                    frame_offset = int((next_file_size - curr_byte_offset) / self.frame_size)
        return None, None



    def create_feature_images(self, feature_df, pano_df, skycam_dir, module_id, verbose=False, allow_skip=True):
        """For each original skycam image:
            1. Get its unix timestamp.
            2. Find the corresponding panoseti image frame, if it exists.
            3. Generate a corresponding set of panoseti image features relative to that frame.
        Note: must download skycam data before calling this routine.
        """
        if module_id not in self.obs_pff_files or len(self.obs_pff_files[module_id]) == 0:
            return feature_df, pano_df
        module_pff_files = self.obs_pff_files[module_id]

        skycam_imgs_root_path = get_skycam_root_path(self.batch_path)
        skycam_subdirs = get_skycam_subdirs(f'{skycam_imgs_root_path}/{skycam_dir}')
        print(f'Generating features for module {module_id}')
        for original_skycam_fname in sorted(os.listdir(skycam_subdirs['original'])):
            if not original_skycam_fname.endswith('.jpg'):
                continue
            if verbose: print(f'\nGenerating pano features for {original_skycam_fname}...')

            # Correlate skycam img to panoseti image, if possible
            t = get_skycam_img_time(original_skycam_fname)
            t = t - timedelta(seconds=60)  # Offset skycam timestamp by typical integration time
            skycam_unix_t = get_unix_from_datetime(t)
            skycam_uid = get_skycam_uid(original_skycam_fname)
            pano_frame_seek_info = self.module_file_time_seek(module_id, skycam_unix_t)
            if pano_frame_seek_info is None:
                if verbose: print('Failed to find matching panoseti frames. Skipping...')
                continue

            # Generate all features
            figs = dict()
            figs['original'] = self.make_original_fig(
                pano_frame_seek_info['file_idx'],
                pano_frame_seek_info['frame_offset'],
                module_id,
                vmin=30,#-3.5,
                vmax=282,#3.5,
                cmap='mako',
                verbose=verbose
            )
            figs['fft'] = self.make_fft_fig(
                pano_frame_seek_info['file_idx'],
                pano_frame_seek_info['frame_offset'],
                module_id,
                vmin=3,
                vmax=10,
                cmap='icefire',
                verbose=verbose
            )
            figs['derivative'], figs['fft-derivative'] = self.make_time_derivative_figs(
                pano_frame_seek_info['file_idx'],
                pano_frame_seek_info['frame_offset'],
                module_id,
                20,
                60,
                ncols=3,
                vmin=[-3, -1],
                vmax=[3, 6],
                cmap=["icefire", "icefire"],
                verbose=verbose
            )

            # Check if all figs are valid
            all_figs_valid = True
            for img_type, fig in figs.items():
                if fig is None:
                    all_figs_valid = False
                    msg = f'The following frame resulted in a None "{img_type}" figure: {pano_frame_seek_info}.'
                    if not allow_skip:
                        raise ValueError(msg)
                    if verbose: print(msg)

            # Skip this image if not all figs are valid
            if not all_figs_valid:
                for fig in figs.values():
                    plt.close(fig)
                if verbose: print('Failed to create figures')
                continue

            # Write figures to data dirs
            pano_fname = module_pff_files[pano_frame_seek_info['file_idx']]['fname']
            frame_offset = pano_frame_seek_info['frame_offset']
            pano_uid = get_pano_uid(pano_fname, frame_offset)
            for img_type, fig in figs.items():
                if verbose: print(f"Creating {get_pano_img_path(self.pano_path, pano_uid, img_type)}")
                fig.savefig(get_pano_img_path(self.pano_path, pano_uid, img_type))
                plt.close(fig)

            # Update dataframes
            pano_df = add_pano_img(
                pano_df,
                pano_uid,
                self.run_dir,
                pano_fname,
                frame_offset,
                module_id,
                pano_frame_seek_info['frame_unix_t'],
                self.batch_id
            )
            feature_df = add_feature_entry(
                feature_df,
                skycam_uid,
                pano_uid,
                self.batch_id
            )
        return feature_df, pano_df


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

def plot_image_fft(data, cmap, **kwargs):
    if data is None or not isinstance(data, np.ndarray):
        print('no image')
        return None
    if data.shape != (32, 32):
        data = np.reshape(data, (32, 32))

    # window image to improve fft
    # shape = kwargs.get("shape", data.shape)
    shape = data.shape
    window_type = "cosine"
    data = data * window(window_type, shape)

    # perform fft and get absolute value
    data = np.abs(fftn(data))
    # shift the DC component to center
    data = fftshift(data)
    data = np.log(data)
    # print(f'min:{np.min(data)}, max:{np.max(data)}')


    ax = isns.imgplot(
        data,
        cmap=cmap,
        cbar=False,
        showticks=False,
        describe=False,
        despine=None,
        **kwargs
    )
    return ax.get_figure()
    ax = isns.fftplot(data, cmap=cmap, window_type='cosine')

