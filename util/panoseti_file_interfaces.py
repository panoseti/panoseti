"""
Class interfaces for working with data contained in Panoseti observing runs.

Use these classes to simplify analysis code.
"""
import typing
import os
import json
import sys
from datetime import datetime, timezone, timedelta
from itertools import chain

import numpy as np
import pandas as pd
import seaborn_image as isns
import matplotlib.pyplot as plt

import config_file
import pff
import image_quantiles


class ObservingRunInterface:

    def __init__(self, data_dir, run_dir):
        """File manager interface for a single observing run."""
        # Check data paths
        self.data_dir = data_dir
        self.run_dir = run_dir
        self.run_path = f'{self.data_dir}/{self.run_dir}'
        self.check_paths()

        # Get start and stop times
        parsed_run_name = pff.parse_name(run_dir)
        self.start_utc = parsed_run_name['start'].replace('Z', '') + '+00:00'
        self.start_utc = datetime.fromisoformat(self.start_utc)

        # Unpack relevant data_config attributes
        self.data_config = config_file.get_data_config(self.run_path)
        self.run_type = self.data_config["run_type"]
        self.has_imaging_data = False
        if "image" in self.data_config:
            self.has_imaging_data = True
            self.intgrn_usec = float(self.data_config["image"]["integration_time_usec"])
            self.img_bpp = int(self.data_config["image"]["quabo_sample_size"]) // 8  # Bytes per imaging pixel
            self.image_size = 32
            self.bytes_per_image_frame = self.img_bpp * self.image_size ** 2  # Bytes per image: 1024 pixels, each requiring self.img_bpp bytes
            self.bytes_per_header_and_image_frame = None  # Bytes per frame, including JSON header, imaging data, and delimiters

        self.has_pulse_height = False
        if "pulse_height" in self.data_config:
            self.has_pulse_height = True
            self.ph_bpp = 2 # Default value
            self.bytes_per_header_and_pulse_height_frame = None  # Bytes per frame, including JSON header, data, and delimiters
            self.pe_threshold = self.data_config["pulse_height"]["pe_threshold"]

            self.any_trigger = None
            self.ph_size = 16
            if "any_trigger" in self.data_config["pulse_height"]:
                self.any_trigger = self.data_config["pulse_height"]["any_trigger"]
                if "group_frames" in self.data_config["pulse_height"]["any_trigger"]:
                    self.ph_group_frames = self.data_config["pulse_height"]["any_trigger"]["group_frames"]
                    if self.ph_group_frames:
                        # If anytrigger is on and group_frames is specified, the 4 PH frames produced per event are
                        # grouped together into a single 1024-pixel image.
                        self.bytes_per_ph_frame = self.ph_bpp * 1024
                        self.ph_size = 32

            self.bytes_per_ph_frame = self.ph_bpp * self.ph_size**2  # Bytes per PH image: [256 or 1024] pixels, each requiring self.img_bpp bytes

            self.two_pixel_trigger = False
            if "two_pixel_trigger" in self.data_config["pulse_height"]:
                self.two_pixel_trigger = self.data_config["pulse_height"]["two_pixel_trigger"]

            self.three_pixel_trigger = False
            if "three_pixel_trigger" in self.data_config["pulse_height"]:
                self.three_pixel_trigger = self.data_config["pulse_height"]["three_pixel_trigger"]

        # Create a dict of all valid pff files available for analysis, indexed by module.
        self.obs_config = config_file.get_obs_config(self.run_path)
        self.obs_pff_files = dict()
        for dome in self.obs_config["domes"]:
            for module in dome["modules"]:
                module_id = config_file.ip_addr_to_module_id(module["ip_addr"])
                self.obs_pff_files[module_id] = {
                    "img": [],
                    "ph": []
                }
        if self.has_imaging_data:
            self.check_image_pff_files()
            self.index_image_pff_files()

        if self.has_pulse_height:
            self.check_pulse_height_pff_files()
            self.index_pulse_height_pff_files()

        with (open(f'{self.run_path}/recording_ended', 'r') as f):
            # recording_ended file is in PDT time
            iso_str = f.readline().rstrip('\n') + '-07:00'
            self.stop_utc = datetime.fromisoformat(iso_str).astimezone(timezone.utc)


    def check_paths(self):
        """Check if data_dir and run_dir exist."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"The data_directory '{self.data_dir}' does not exist!")
        elif not os.path.exists(self.run_path):
            raise FileNotFoundError(f"The run directory at '{self.run_path}' does not exist!")

    def check_pulse_height_pff_files(self):
        """Run basic checks on pff data files. Raises an exception if any checks fail."""
        for fname in os.listdir(self.run_path):
            fpath = f'{self.run_path}/{fname}'
            if not (pff.is_pff_file(fname)
                    and pff.pff_file_type(fname) in ['ph256', 'ph1024']
                    and os.path.getsize(fpath) > 0):
                continue
            parsed_name = pff.parse_name(fname)
            module_id = int(parsed_name["module"])
            if module_id not in self.obs_pff_files:
                raise FileExistsError(f'{fpath} was not generated by a module specified in obs_config.\n'
                                      f'This file may have been moved to {self.run_dir} from a different run.')
            if int(parsed_name["bpp"]) != self.ph_bpp:
                raise FileExistsError(f'The bytes per pixel of {fpath} do not match the value specified in data_config.\n'
                                      f'This file may have been moved to {self.run_dir} from a different run.')

    def index_pulse_height_pff_files(self):
        """Indexes all available pulse height files for each module."""
        for fname in os.listdir(self.run_path):
            fpath = f'{self.run_path}/{fname}'
            if not (pff.is_pff_file(fname)
                    and pff.pff_file_type(fname) in ['ph256', 'ph1024']
                    and os.path.getsize(fpath) > 0):
                continue
            parsed_name = pff.parse_name(fname)
            module_id = int(parsed_name["module"])
            attrs = dict()
            attrs["fname"] = fname
            attrs["seqno"] = int(parsed_name["seqno"])
            with open(fpath, 'rb') as f:
                self.bytes_per_header_and_pulse_height_frame, attrs["nframes"], attrs["first_unix_t"], \
                    attrs["last_unix_t"] = pff.img_info(f, self.bytes_per_ph_frame)
            self.obs_pff_files[module_id]["ph"] += [attrs]
        # Sort files_to_process in ascending order by file sequence number
        for module_id in self.obs_pff_files:
            self.obs_pff_files[module_id]["ph"].sort(key=lambda attrs: attrs["seqno"])

    def check_image_pff_files(self):
        """Run basic checks on image pff data files. Raises an exception if any checks fail."""
        for fname in os.listdir(self.run_path):
            fpath = f'{self.run_path}/{fname}'
            if not (pff.is_pff_file(fname)
                    and pff.pff_file_type(fname) in ['img8', 'img16']
                    and os.path.getsize(fpath) > 0):
                continue
            parsed_name = pff.parse_name(fname)
            module_id = int(parsed_name["module"])
            if module_id not in self.obs_pff_files:
                raise FileExistsError(f'{fpath} was not generated by a module specified in obs_config.\n'
                                      f'This file may have been moved to {self.run_dir} from a different run.')
            if int(parsed_name["bpp"]) != self.img_bpp:
                raise FileExistsError(f'The bytes per pixel of {fpath} do not match the value specified in data_config.\n'
                                      f'This file may have been moved to {self.run_dir} from a different run.')

    def index_image_pff_files(self):
        """Indexes all available imaging files for each module.."""
        for fname in os.listdir(self.run_path):
            fpath = f'{self.run_path}/{fname}'
            if not (pff.is_pff_file(fname)
                    and pff.pff_file_type(fname) in ['img16', 'img8']
                    and os.path.getsize(fpath) > 0):
                continue
            parsed_name = pff.parse_name(fname)
            module_id = int(parsed_name["module"])
            attrs = dict()
            attrs["fname"] = fname
            attrs["seqno"] = int(parsed_name["seqno"])
            with open(fpath, 'rb') as f:
                self.bytes_per_header_and_image_frame, attrs["nframes"], attrs["first_unix_t"], \
                    attrs["last_unix_t"] = pff.img_info(f, self.bytes_per_image_frame)
            self.obs_pff_files[module_id]["img"] += [attrs]
        # Sort files_to_process in ascending order by file sequence number
        for module_id in self.obs_pff_files:
            self.obs_pff_files[module_id]["img"].sort(key=lambda attrs: attrs["seqno"])

    @staticmethod
    def read_frame(f, bytes_per_pixel, allow_partial_images=False, frame_size=32):
        """Returns the next frame and json header from f. If at EOF, returns (None, None)."""
        assert frame_size in [16, 32], 'Unrecognized frame_size: {0}'.format(frame_size)
        j, img = None, None
        json_str = pff.read_json(f)
        if json_str is not None:
            j = json.loads(json_str)
            # Screen for partial quabo images
            if not allow_partial_images:
                if frame_size == 32:
                    for quabo in j:
                        # TODO: verify that this actually works.
                        if j[quabo]['tv_sec'] == 0:  # tv_sec is 0 iff the DAQ node received no data for a quabo.
                            return None, None
                elif frame_size == 16:
                    if j['tv_sec'] == 0:
                        return None, None
            img = pff.read_image(f, frame_size, bytes_per_pixel)
            img = np.array(img)
            img = np.reshape(img, (frame_size, frame_size))
        return j, img

    # @staticmethod
    # def read_pulse_height_frame(f, bytes_per_pixel, allow_partial_images=False, ph_type='ph256'):
    #     """Returns the next image frame and json header from f. If at EOF, returns (None, None)."""
    #     if ph_type == 'ph1024':
    #         ObservingRunInterface.read_image_frame(f, bytes_per_pixel, allow_partial_images)
    #     elif ph_type == 'ph256':
    #         j, img = None, None
    #         json_str = pff.read_json(f)
    #         if json_str is not None:
    #             j = json.loads(json_str)
    #             # Screen for partial quabo images
    #             if not allow_partial_images and j['tv_sec'] == 0:
    #                 return None, None
    #             img = pff.read_image(f, 32, bytes_per_pixel)
    #             img = np.array(img)
    #             img = np.reshape(img, (32, 32))
    #         return j, img
    #     else:
    #         raise ValueError("Unrecognized ph_type '{0}'".format(ph_type))
    #
    #

    @staticmethod
    def plot_image(img, **kwargs):
        if img is None or not isinstance(img, np.ndarray):
            print('no image')
            return None
        if img.shape != (32, 32):
            img = np.reshape(img, (32, 32))
        ax = isns.imghist(img, **kwargs)  # vmin=vmin, vmax=vmax, robust=True)#vmin=max(0, mean - 2.5 * std), vmax=mean + 2.5 * std)
        # ax = isns.imghist(img, cmap="viridis", vmin=-100, vmax=100)#vmin=max(0, mean - 2.5 * std), vmax=mean + 2.5 * std)
        # ax = isns.imghist(img, cmap="viridis", vmin=-3.5, vmax=3.5)#vmin=max(0, mean - 2.5 * std), vmax=mean + 2.5 * std)
        return ax.get_figure()

    @staticmethod
    def get_tz_timestamp_str(unix_t, tz_hr_offset=0):
        """Returns the timestamp string, offset from utc by tz_hr_offset hours."""
        dt = datetime.fromtimestamp(unix_t, timezone(timedelta(hours=tz_hr_offset)))
        return dt.strftime("%m/%d/%Y, %H:%M:%S")


    def stack_frames(self, start_file_idx, start_frame_offset, module_id, delta_t=1, agg='sum', allow_partial_image=False):
        """
        Evenly samples image frames between now and now-delta_t, then aggregates
        the frames according to the given aggregation method.

        By default, stack until a total of 6ms of observational data is accumulated. e.g.:
        - if you have 100us panoseti image data, add 60 images together, roughly evenly spaced over 1 second.
        - if you have 2ms panoseti image data, add 3 images together, roughly evenly spaced over 1 second.
        - if you have 1ms panoseti image data, add 6 images together, roughly evenly spaced over 1 second.

        @param f: file pointer to a PFF imaging file
        @param nframes: number of frames to stack
        @param delta_t: seconds between current and oldest frame to be sampled
        @param agg: aggregation method
        @return: Stacked image frame
        """

        STACKED_INTEGRATION_MS = 6  # Stack 6ms of image frame data by default. e.g.:

        nframes = int((STACKED_INTEGRATION_MS * 1E-3) / (self.intgrn_usec * 1E-6))
        module_image_pff_files = self.obs_pff_files[module_id]["img"]
        time_step = delta_t / nframes   # time step between sampled frames
        frame_step_size = int(time_step / (self.intgrn_usec * 1E-6))
        assert frame_step_size > 0
        assert delta_t >= 0, 'stack_frames is a causal function.'
        # Iterate backwards through PFF files for module_id
        frame_buffer = np.zeros((nframes, 32, 32))
        frame_offset = start_frame_offset
        n = 0
        for i in range(start_file_idx, -1, -1):
            if n == nframes: break
            file_info = module_image_pff_files[i]
            fpath = f"{self.run_path}/{file_info['fname']}"
            with open(fpath, 'rb') as fp:
                # Start file pointer with an offset based on the previous file -> ensures even frame sampling
                fp.seek(frame_offset * self.bytes_per_header_and_image_frame, os.SEEK_CUR)
                # Create FrameIterator iterator
                fitr = self.image_frame_iterator(fp, frame_step_size, nframes - n)
                for j, img in fitr:
                    frame_buffer[n] = img
                    n += 1
                # Get info for next file if we need more frames.
                if i > 0:
                    next_file_size = module_image_pff_files[i - 1]['nframes'] * self.bytes_per_header_and_image_frame
                    curr_byte_offset = frame_step_size * self.bytes_per_header_and_image_frame - fp.tell()
                    frame_offset = int((next_file_size - curr_byte_offset) / self.bytes_per_header_and_image_frame)
        if len(frame_buffer) < nframes:
            raise ValueError(f'Insufficient frames for frame stacking: '
                             f'retrieved {len(frame_buffer)} / {nframes} frames')
        if agg == 'mean':
            return np.mean(frame_buffer, axis=0)
        elif agg == 'sum':
            return np.sum(frame_buffer, axis=0)

    def compute_spatial_median(self, img_stack):
        """
        Given an image stack (n, 32,32) of n module images, returns a single (32,32) array containing
        the 16 median 8x8 pixel region of the n images.
        """
        if not isinstance(img_stack, np.ndarray):
            raise ValueError(f'module image must be an np.ndarray, got {type(img_stack)}')
        spatial_median = np.zeros((32, 32))
        for i in range(0, 32, 8):
            for j in range(0, 32, 8):
                raw_med = np.median(img_stack[:, i:i + 8, j:j + 8], axis=0)
                std = np.std(raw_med)
                mu = np.median(raw_med)
                # spatial_median[i:i + 8, j:j + 8] = np.clip(raw_med, mu - std, mu + std)
                spatial_median[i:i + 8, j:j + 8] = np.median(img_stack[:, i:i + 8, j:j + 8])
        return spatial_median


    def compute_module_supermedian_image(self, module_id, spatial_median_window_usec=30 *10**6, max_samples_per_window=500):
        module_image_files = self.obs_pff_files[module_id]["img"]
        buffer = []
        # First pass: Sample the night of data and remove spatial medians at 1s intervals.
        available_images_per_window = spatial_median_window_usec / self.intgrn_usec
        frames_per_window = int(min(available_images_per_window, max_samples_per_window))
        frame_step = int(max(1, available_images_per_window / frames_per_window))
        for img_file in module_image_files:
            fname = img_file["fname"]
            fpath = f'{self.run_path}/{fname}'
            with open(fpath, 'rb') as fp:
                frame_iterator = self.image_frame_iterator(fp, step_size=frame_step, frame_limit=None)
                for j, img in frame_iterator:
                    buffer.append(img)
        nwindows = len(buffer) // frames_per_window
        trimmed_len = nwindows * frames_per_window
        buffer = np.array(buffer[0:trimmed_len])
        buffer_no_spatial_medians = np.zeros((buffer.shape))
        # buffer = buffer.reshape((frames_per_window, nwindows, 32, 32))
        spatial_medians = np.zeros((nwindows, 32, 32))
        for i in range(0, nwindows):
            l = i * frames_per_window
            r = (i + 1) * frames_per_window
            spatial_medians[i] = self.compute_spatial_median(buffer[l:r])
            buffer_no_spatial_medians[l:r] = buffer[l:r] - spatial_medians[i]
        expanded_spatial_medians = np.tile(spatial_medians, (frames_per_window, 1, 1))
        supermedian = np.median(buffer_no_spatial_medians, axis=0)
        flat = buffer - expanded_spatial_medians - supermedian
        return spatial_medians, buffer, buffer_no_spatial_medians, supermedian, flat
        # return spatial_medians
        # Second pass: Compute medians for each pixel across the entire night.
        return



    def module_file_time_seek(self, module_id, target_time):
        """Search module data to find the image frame with timestamp closest to target_time.
        target_time should be a unix timestamp."""
        module_image_pff_files = self.obs_pff_files[module_id]["img"]
        # Use binary search to find the file that contains target_time
        l = 0
        r = len(module_image_pff_files) - 1
        m = -1
        while l <= r:
            m = (r + l) // 2
            file_info = module_image_pff_files[m]
            if target_time > file_info['last_unix_t']:
                l = m + 1
            elif target_time < file_info['first_unix_t']:
                r = m - 1
            else:
                break
        file_info = module_image_pff_files[m]
        if target_time < file_info['first_unix_t'] or target_time > file_info['last_unix_t']:
            return None
        # Use binary search to find the frame closest to target_time
        fpath = f"{self.run_path}/{file_info['fname']}"
        with open(fpath, 'rb') as fp:
            frame_time = self.intgrn_usec * 10 ** (-6)
            pff.time_seek(fp, frame_time, self.bytes_per_image_frame, target_time)
            frame_offset = int(fp.tell() / self.bytes_per_header_and_image_frame)
            j, img = self.read_frame(fp, self.img_bpp)
            if j is None or img is None:
                return None
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


    def image_frame_iterator(self, fp, step_size: int, frame_limit=None):
        """
        @param fp: file pointer to an image PFF file
        @param step_size: frame step size
        @param frame_limit: upper limit on number of frames to return.
        @return: PFFFrameIterator that returns frames from PFF file fp in step_size frame increments.
        """
        return PFFFrameIterator(
            fp, step_size, self.bytes_per_header_and_image_frame, self.img_bpp, frame_limit, self.image_size
        )

    def pulse_height_frame_iterator(self, fp, step_size: int, frame_limit=None):
        """
        @param fp: file pointer to a pulse-height PFF file
        @param step_size: frame step size
        @param frame_limit: upper limit on number of frames to return.
        @return: PFFFrameIterator that returns frames from PFF file fp in step_size frame increments.
        """
        return PFFFrameIterator(
            fp, step_size, self.bytes_per_header_and_pulse_height_frame, self.ph_bpp, frame_limit, self.ph_size
        )


class PFFFrameIterator:
    def __init__(self, fp, step_size, bytes_per_header_and_frame, bytes_per_pixel, frame_limit, frame_size):
        """From the PFF file pointer fp, returns the image frame
        and json header step_size later than the previous image.
        Note that while imaging frames are typically in chronological order,
        this is not a guarantee.
        @param fp: file pointer to a PFF file
        @param step_size: frame step size
        """
        self.fp = fp
        self.bytes_per_frame = bytes_per_header_and_frame
        self.bpp = bytes_per_pixel
        self.step_size = step_size
        self.iter_num = 0
        self.frame_limit = frame_limit
        self.frame_size = frame_size

    def __iter__(self):
        return self

    def __next__(self):
        if (self.frame_limit is not None) and (self.iter_num >= self.frame_limit):
            raise StopIteration
        elif self.iter_num >= 1:
            seek_dist = (self.step_size - 1) * self.bytes_per_frame
            if seek_dist < 0 and (self.fp.tell() + seek_dist < 0):
                raise StopIteration
            self.fp.seek(seek_dist, os.SEEK_CUR)  # Skip (step_size - 1) images
        self.iter_num += 1
        j, img = ObservingRunInterface.read_frame(self.fp, self.bpp, frame_size=self.frame_size)
        if (j is None) or (img is None):
            raise StopIteration
        return j, img

if __name__ == '__main__':
    # data_dir = '/Users/nico/Downloads/panoseti_test_data/obs_data/data'
    # run_dir = 'obs_Lick.start_2023-08-29T04:49:58Z.runtype_sci-obs.pffd'

    data_dir = '/Users/nico/Downloads/panoseti_test_data/obs_data/data'
    run_dir = 'obs_Lick.start_2023-08-01T05:14:21Z.runtype_sci-obs.pffd'

    ori = ObservingRunInterface(data_dir, run_dir)

    ori.compute_module_supermedian_image(3)
