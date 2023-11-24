"""
PanosetiBatchBuilder creates all panoseti features for a single observing run.
"""
import os
import numpy as np
import sys
from datetime import timedelta

from batch_building_utils import *
from panoseti_file_interfaces import ObservingRunInterface
from skycam_utils import get_skycam_img_time
from pano_utils import *
from dataframe_utils import *
from dataset_utils import PanoDatasetBuilder


sys.path.append("../../util")
import pff

class PanoBatchBuilder(ObservingRunInterface):

    def __init__(self, data_dir, run_dir, task, batch_id, force_recreate=False, verbose=False):
        super().__init__(data_dir, run_dir)
        self.task = task
        self.batch_id = batch_id
        self.batch_dir = get_batch_dir(task, batch_id)
        self.batch_path = f'{batch_data_root_dir}/{self.batch_dir}'
        self.pano_path = f'{self.batch_path}/{pano_imgs_root_dir}/{run_dir}'
        self.pano_subdirs = get_pano_subdirs(self.pano_path)
        self.force_recreate = force_recreate
        self.verbose = verbose
        self.pano_dataset_builder = PanoDatasetBuilder(task, batch_id, self.run_dir)

        self.init_pano_subdirs()

    def init_pano_subdirs(self):
        """Initialize pre-processing directories."""
        for dir_name in self.pano_subdirs.values():
            os.makedirs(dir_name, exist_ok=True)
        # if V
        # self.is_uninitialized()

    """State checks"""

    def is_uninitialized(self):
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

    def pano_features_created(self):
        """Returns True iff skycam features are already created."""
        ret = False
        if os.path.exists(f'{self.batch_path}/{pano_path_index_fname}'):
            ret = True
        for pano_subdir in self.pano_subdirs.values():
            ret &= os.path.exists(pano_subdir) and len(os.listdir(pano_subdir)) > 0
        return ret

    """Feature creation"""

    def img_transform(self, img):
        img = np.reshape(img, (32, 32))
        # img = np.flip(img, axis=0)
        img = np.rot90(img, 2)
        return img

    def make_original_fig(self, start_file_idx, start_frame_offset, module_id, vmin, vmax, cmap):
        module_pff_files = self.obs_pff_files[module_id]
        file_info = module_pff_files[start_file_idx]
        fpath = f"{self.run_path}/{file_info['fname']}"
        with open(fpath, 'rb') as fp:
            fp.seek(
                start_frame_offset * self.frame_size,
                os.SEEK_CUR
            )
            j, img = self.read_image_frame(fp, self.img_bpp)
            self.pano_dataset_builder.add_img_to_entry(img, 'original', self.img_bpp)
            img = self.img_transform(img)
            #img = (img - np.median(img)) / np.std(img)
            fig = self.plot_image(img, vmin=vmin, vmax=vmax, bins=40, cmap=cmap, perc=(0.5, 99.5))
            #plt.pause(0.5)
            return fig

    def make_fft_fig(self, start_file_idx, start_frame_offset, module_id, vmin, vmax, cmap):
        module_pff_files = self.obs_pff_files[module_id]
        file_info = module_pff_files[start_file_idx]
        fpath = f"{self.run_path}/{file_info['fname']}"
        with open(fpath, 'rb') as fp:
            fp.seek(
                start_frame_offset * self.frame_size,
                os.SEEK_CUR
            )
            j, img = self.read_image_frame(fp, self.img_bpp)
            self.pano_dataset_builder.add_img_to_entry(apply_fft(img), 'fft', self.img_bpp)
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
                                  cmap):
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
            j, img = self.read_image_frame(f, self.img_bpp)
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
            if self.verbose: print(f"Processing {file_info['fname']}")
            with open(fpath, 'rb') as fp:
                if self.verbose: print("newfile")
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
                        delta_t = -step_delta_t * k
                        delta_ts.append(str(delta_t))
                        diff = (img - hist[k - 1])
                        # if delta_t == -60:
                        #     self.img_array_builder.add_img_to_entry(diff, 'derivative-60s')
                        data = diff / np.std(hist)
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

    def correlate_skycam_to_pano_img(self, skycam_unix_t, module_id):
        # Correlate skycam img to panoseti image, if possible
        # t = get_skycam_img_time(original_skycam_fname)
        # t = skycam_unix_t - timedelta(seconds=60)  # Offset skycam timestamp by typical integration time
        # skycam_unix_t = get_unix_from_datetime(t)
        skycam_integration_offset = -60
        return self.module_file_time_seek(module_id, skycam_unix_t + skycam_integration_offset)

    def create_feature_images(self, feature_df, pano_df, module_id, skycam_df, skycam_dir, allow_skip=True):
        """For each original skycam image:
            1. Get its unix timestamp.
            2. Find the corresponding panoseti image frame, if it exists.
            3. Generate a corresponding set of panoseti image features relative to that frame.
        Note: must download skycam data before calling this routine.
        """

        if self.pano_features_created() and not self.force_recreate:
            raise FileExistsError(f"Data in {self.run_dir} already processed")

        module_pff_files = self.obs_pff_files[module_id]

        print(f'Generating features for module {module_id}')
        skycam_info = skycam_df.loc[skycam_df.skycam_dir == skycam_dir, ['skycam_uid', 'unix_t']]
        for index, skycam_row in skycam_info.sort_values(by='unix_t').iterrows():
            skycam_uid, skycam_unix_t = skycam_row
            if self.verbose: print(f'\nGenerating pano features for skycam_uid {skycam_uid}...')
            pano_frame_seek_info = self.correlate_skycam_to_pano_img(skycam_unix_t, module_id)
            if pano_frame_seek_info is None:
                if self.verbose: print('Failed to find matching panoseti frames. Skipping...')
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
            )
            figs['fft'] = self.make_fft_fig(
                pano_frame_seek_info['file_idx'],
                pano_frame_seek_info['frame_offset'],
                module_id,
                vmin=3,
                vmax=10,
                cmap='icefire',
            )
            figs['derivative'], figs['fft-derivative'] = self.make_time_derivative_figs(
                pano_frame_seek_info['file_idx'],
                pano_frame_seek_info['frame_offset'],
                module_id,
                2,
                60,
                ncols=3,
                vmin=[-3, -1],
                vmax=[3, 6],
                cmap=["icefire", "icefire"],
            )

            # Check if all figs are valid
            all_figs_valid = True
            for img_type, fig in figs.items():
                if fig is None:
                    all_figs_valid = False
                    msg = f'The following frame resulted in a None "{img_type}" figure: {pano_frame_seek_info}.'
                    if not allow_skip:
                        raise ValueError(msg)
                    if self.verbose: print(msg)

            # Skip this image if not all figs are valid
            if not all_figs_valid:
                for fig in figs.values():
                    plt.close(fig)
                if self.verbose: print('Failed to create figures')
                self.pano_dataset_builder.clear_current_entry()
                continue

            # Write figures to data dirs
            pano_fname = module_pff_files[pano_frame_seek_info['file_idx']]['fname']
            frame_offset = pano_frame_seek_info['frame_offset']
            pano_uid = get_pano_uid(pano_fname, frame_offset)
            for img_type, fig in figs.items():
                if self.verbose: print(f"Creating {get_pano_img_path(self.pano_path, pano_uid, img_type)}")
                fig.savefig(get_pano_img_path(self.pano_path, pano_uid, img_type))
                plt.close(fig)

            # Commit entry to data_array for this run_dir
            self.pano_dataset_builder.write_arrays(pano_uid)

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

    def build_pano_batch_data(self, feature_df, pano_df, skycam_df, skycam_dir):
        print(f'Creating panoseti features for {self.run_dir}')
        for module_id in self.obs_pff_files:
            if len(self.obs_pff_files[module_id]) == 0:
                continue
            feature_df, pano_df = self.create_feature_images(
                feature_df, pano_df, module_id, skycam_df, skycam_dir
            )
        return feature_df, pano_df

