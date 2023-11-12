
import os
import numpy as np
import sys
from collections import deque
import seaborn as sns

import matplotlib.pyplot as plt

from panoseti_file_interfaces import ObservingRunFileInterface, ModuleImageInterface
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
            self.is_data_preprocessed()
        except FileExistsError as ferr:
            if not self.force_recreate:
                raise ferr
        for dir_name in self.pano_subdirs.values():
            os.makedirs(dir_name, exist_ok=True)

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
        if os.path.exists(self.pano_subdirs['derivative']) and len(os.listdir(self.pano_subdirs['derivative'])) > 0:
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
            img = (img - np.median(img)) / np.std(img)
            fig = self.plot_image(img, vmin=vmin, vmax=vmax, cmap=cmap)
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
            fig = plot_image_fft(img, vmin=vmin, vmax=vmax, cmap=cmap)
            return fig

    def make_time_derivative_figs(self,
                                  start_file_idx,
                                  start_frame_offset,
                                  module_id,
                                  step_delta_t,
                                  max_delta_t,
                                  nrows,
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
        @param nrows: number of evenly spaced time-derivatives
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
                    for k in [int(i * hist_size / nrows) for i in range(nrows, 0, -1)]:
                        delta_ts.append(str(-step_delta_t * k))
                        data = (img - hist[k - 1]) / np.std(hist)
                        imgs.append(data)
                    # print(delta_ts)

                    fig_time_derivative = plot_time_derivative(
                        imgs, delta_ts, vmin=vmin[0], vmax=vmax[0], cmap=cmap[0]
                    )
                    fig_fft_time_derivative = plot_fft_time_derivative(
                        imgs, delta_ts, 4, vmin[1], vmax[1], cmap=cmap[1]
                    )

                    # plt.pause(0.5)
                    return fig_time_derivative, fig_fft_time_derivative
                # Compute the frame offset for the next pff file
                if i > 0:
                    next_file_size = module_pff_files[i-1]['nframes'] * self.frame_size
                    curr_byte_offset = frame_step_size * self.frame_size - fp.tell()
                    frame_offset = int((next_file_size - curr_byte_offset) / self.frame_size)

    def create_feature_images(self, feature_df, pano_df, skycam_dir, verbose=False, allow_skip=True):
        """For each original skycam image:
            1. Get its unix timestamp.
            2. Find the corresponding panoseti image frame, if it exists.
            3. Generate a corresponding set of panoseti image features relative to that frame.
        Note: must download skycam data before calling this routine.
        """
        skycam_imgs_root_path = get_skycam_root_path(self.batch_path)
        skycam_subdirs = get_skycam_subdirs(f'{skycam_imgs_root_path}/{skycam_dir}')

        module_id = 254
        self.init_preprocessing_dirs()
        for original_skycam_fname in sorted(os.listdir(skycam_subdirs['original'])):
            if original_skycam_fname.endswith('.jpg'):
                if verbose: print(f'\nGenerating pano features for {original_skycam_fname}...')
                # Correlate skycam img to panoseti image, if possible
                t = get_skycam_img_time(original_skycam_fname)
                skycam_unix_t = get_unix_from_datetime(t)
                skycam_uid = get_skycam_uid(original_skycam_fname)
                pano_frame_info = self.module_file_time_seek(module_id, skycam_unix_t)

                if pano_frame_info is not None:
                    # Generate all features
                    figs = dict()
                    figs['original'] = self.make_original_fig(
                        pano_frame_info['file_idx'],
                        pano_frame_info['frame_offset'],
                        module_id,
                        vmin=-3.5,
                        vmax=3.5,
                        cmap='mako',
                        verbose=verbose
                    )
                    figs['fft'] = self.make_fft_fig(
                        pano_frame_info['file_idx'],
                        pano_frame_info['frame_offset'],
                        module_id,
                        vmin=3,
                        vmax=10,
                        cmap='icefire',
                        verbose=verbose
                    )
                    figs['derivative'], figs['fft-derivative'] = self.make_time_derivative_figs(
                        pano_frame_info['file_idx'],
                        pano_frame_info['frame_offset'],
                        module_id,
                        1,
                        60,
                        nrows=4,
                        vmin=[-3.5, -1],
                        vmax=[3.5, 6],
                        cmap=["icefire", "icefire"],
                        verbose=verbose
                    )
                    # Check if all figs are valid
                    all_figs_valid = True
                    for img_type, fig in figs.items():
                        if fig is None:
                            all_figs_valid = False
                            msg = f'The following frame resulted in a None "{img_type}" figure: {pano_frame_info}.'
                            if not allow_skip:
                                raise ValueError(msg)
                            if verbose: print(msg)
                    # Skip this image if not all figs are valid
                    if not all_figs_valid:
                        for fig in figs.values():
                            plt.close(fig)
                        continue
                    # Write figures to data dirs
                    # plt.pause(.5)
                    module_pff_files = self.obs_pff_files[module_id]
                    pano_fname = module_pff_files[pano_frame_info['file_idx']]['fname']
                    pano_uid = get_pano_uid(skycam_uid, pano_fname)
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
                        module_id,
                        unix_t=pano_frame_info['frame_unix_t'],
                        batch_id=self.batch_id
                    )
                    feature_df = add_feature_entry(
                        feature_df,
                        skycam_uid,
                        pano_uid,
                        self.batch_id
                    )
        # for img_type, fig in figs.items():
        #     plt.close(fig)

        # plt.pause(1)
        # plt.close(fig)
        # file_info = self.obs_pff_files[module_id][pano_frame_info['file_idx']]
        # # print(builder.start_utc <= t <= builder.stop_utc)
        # print('delta_t = ', (file_info['last_unix_t'] - file_info['first_unix_t']))
        # print(file_info['seqno'])
        # print()
        #
        return feature_df, pano_df

    def add_pano_data_to_pano_df(self, pano_df, batch_id, verbose):
        """Add entries for each pano image to pano_df """
        pass
