"""
Class interfaces for working with data contained in Panoseti observing runs.

Use these classes to simplify analysis code.
"""

import os
import json
import sys
from datetime import datetime, timezone, timedelta
from itertools import chain

import numpy as np
import seaborn_image as isns
import matplotlib.pyplot as plt

sys.path.append("../../util")
import config_file
import pff
import image_quantiles


class ObservingRunProxy:

    def __init__(self, data_dir, run_dir):
        """File manager interface for a single observing run."""

        # Check data paths
        self.data_dir = data_dir
        self.run_dir = run_dir
        self.run_path = f'{self.data_dir}/{self.run_dir}'
        self.check_paths()

        # Unpack relevant data_config attributes
        self.data_config = config_file.get_data_config(self.run_path)
        self.run_type = self.data_config["run_type"]
        self.has_imaging_data = False
        if "image" in self.data_config:
            self.has_imaging_data = True
            self.intgrn_usec = float(self.data_config["image"]["integration_time_usec"])
            self.img_bpp = int(self.data_config["image"]["quabo_sample_size"]) // 8  # Bytes per imaging pixel
            self.img_size = self.img_bpp * 32 ** 2
            self.frame_size = None
        self.has_pulse_height = False
        if "pulse_height" in self.data_config:
            self.has_pulse_height = True
            # TODO

        # Create a dict of all valid imaging pff files available for analysis, indexed by module.
        self.obs_config = config_file.get_obs_config(self.run_path)
        self.obs_pff_files = dict()
        for dome in self.obs_config["domes"]:
            for module in dome["modules"]:
                module_id = config_file.ip_addr_to_module_id(module["ip_addr"])
                self.obs_pff_files[module_id] = []
        if self.has_imaging_data:
            self.check_imaging_files()
            self.get_module_imaging_files()

        # Get start and stop times
        parsed_run_name = pff.parse_name(run_dir)
        self.start_utc = parsed_run_name['start'].replace('Z', '') + '+00:00'
        self.start_utc = datetime.fromisoformat(self.start_utc)

        with (open(f'{self.run_path}/recording_ended', 'r') as f):
            # recording_ended file is in PDT time
            iso_str = f.readline().rstrip('\n') + '-07:00'
            self.stop_utc = datetime.fromisoformat(iso_str).astimezone(timezone.utc)

    def read_frame(self, f, bytes_per_pixel):
        """Returns the next image frame and json header from f."""
        j, img = None, None
        json_str = pff.read_json(f)
        if json_str is not None:
            j = json.loads(json_str)
            img = pff.read_image(f, 32, bytes_per_pixel)
            img = np.array(img)
        return j, img

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

    @staticmethod
    def plot_image(img, **kwargs):
        if img is None or not isinstance(img, np.ndarray):
            print('no image')
            return None
        if img.shape != (32, 32):
            img = np.reshape(img, (32, 32))
        ax = isns.imghist(img, **kwargs)#vmin=vmin, vmax=vmax, robust=True)#vmin=max(0, mean - 2.5 * std), vmax=mean + 2.5 * std)
        # ax = isns.imghist(img, cmap="viridis", vmin=-100, vmax=100)#vmin=max(0, mean - 2.5 * std), vmax=mean + 2.5 * std)
        # ax = isns.imghist(img, cmap="viridis", vmin=-3.5, vmax=3.5)#vmin=max(0, mean - 2.5 * std), vmax=mean + 2.5 * std)
        return ax.get_figure()

    @staticmethod
    def get_tz_timestamp_str(unix_t, tz_hr_offset=0):
        """Returns the timestamp string, offset from utc by tz_hr_offset hours."""
        dt = datetime.fromtimestamp(unix_t, timezone(timedelta(hours=tz_hr_offset)))
        return dt.strftime("%m/%d/%Y, %H:%M:%S")

    def check_paths(self):
        """Check if data_dir and run_dir exist."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"The data_directory '{self.data_dir}' does not exist!")
        elif not os.path.exists(self.run_path):
            raise FileNotFoundError(f"The run directory at '{self.run_path}' does not exist!")

    def check_imaging_files(self):
        for fname in os.listdir(self.run_path):
            fpath = f'{self.run_path}/{fname}'
            if not (pff.is_pff_file(fname)
                    and pff.pff_file_type(fname) in ['img16', 'img8']
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

    def get_module_imaging_files(self):
        """Returns an array of dictionaries storing info for all available pff files for module_id."""
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
                self.frame_size, attrs["nframes"], attrs["first_unix_t"], \
                    attrs["last_unix_t"] = pff.img_info(f, self.img_size)
            self.obs_pff_files[module_id] += [attrs]
        # Sort files_to_process in ascending order by file sequence number
        for module_id in self.obs_pff_files:
            self.obs_pff_files[module_id].sort(key=lambda attrs: attrs["seqno"])

    def frame_iterator(self, fp, step_size):
        return FrameIterator(fp, step_size, self.frame_size, self.img_bpp)


class FrameIterator:
    def __init__(self, fp, step_size, frame_size, bytes_per_pixel):
        """From the PFF file pointer fp, returns the image frame
        and json header step_size later than the previous image.
        Note that while imaging frames are typically in chronological order,
        this is not a guarantee."""
        self.fp = fp
        self.frame_size = frame_size
        self.bpp = bytes_per_pixel
        self.step_size = step_size
        self.first_itr = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.first_itr:
            seek_dist = (self.step_size - 1) * self.frame_size
            if seek_dist < 0 and (self.fp.tell() + seek_dist < 0):
                raise StopIteration
            self.fp.seek(seek_dist, os.SEEK_CUR)  # Skip (step_size - 1) images
        self.first_itr = False
        j, img = ObservingRunProxy.read_frame(self.fp, self.bpp)
        if img is None:
            raise StopIteration
        return j, img


class ModuleImageInterface(ObservingRunProxy):

    def __init__(self, data_dir, run_dir, module_id, require_data=False):
        """Interface for doing analysis work with images produced by the specified module
        during the given observing run."""
        super().__init__(data_dir, run_dir)
        self.module_id = module_id
        if module_id not in self.obs_pff_files:
            raise ValueError(f"Module {module_id} did not exist for the run '{run_dir}'.")
        self.module_pff_files = self.obs_pff_files[module_id]

        # Check if there are any images for this module.
        if len(self.module_pff_files) == 0:
            msg = f"No valid panoseti imaging data for module {module_id} in '{self.run_dir}'."
            if require_data:
                raise FileNotFoundError(msg)
            else:
                print(msg)
                self.start_unix_t = None
        else:
            self.start_unix_t = self.module_pff_files[0]['first_unix_t']



# class ModuleImageIterator:
#     def __init__(self, module_image_interface, step_size, frame_size, bytes_per_pixel):
#         """Returns all available data from the """
#         assert isinstance(module_image_interface, ModuleImageInterface)
#         self.module_image_interface = module_image_interface
#         self.step_size = step_size
#         self.frame_size = frame_size
#         self.bytes_per_pixel = bytes_per_pixel
#
#
#     def __iter__(self):
#
#         return self
#
#     def __next__(self):
#         m = self.module_image_interface
#         j, img = m.get_next_frame(m.fp, self.step_size, m.frame_size, m.img_bpp)
#         if img is None:
#             raise StopIteration
#         return j, img
#
