
import os
import numpy as np
import sys

import matplotlib.pyplot as plt

from panoseti_file_interfaces import ObservingRunFileInterface, ModuleImageInterface
from skycam_utils import get_batch_dir
from panoseti_batch_utils import *
from dataframe_utils import *

sys.path.append("../../util")
import pff


class PanosetiBatchBuilder(ObservingRunFileInterface):

    def __init__(self, data_dir, run_dir, task, batch_id):
        super().__init__(data_dir, run_dir)
        self.task = task
        self.batch_id = batch_id
        self.batch_dir = get_batch_dir(task, batch_id)
        self.batch_path = f'{batch_data_root_dir}/{self.batch_dir}'
        self.pano_path = f'{self.batch_path}/{pano_imgs_root_dir}/{run_dir}'
        self.pano_subdirs = get_pano_subdirs(self.pano_path)
        os.makedirs(self.pano_path, exist_ok=True)

    def init_preprocessing_dirs(self):
        """Initialize pre-processing directories."""
        self.is_data_preprocessed()
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
            ret = {
                'file_idx': m,
                'offset': fp.tell()
            }
            return ret

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
                    j, img = self.get_next_frame(fp, self.frame_size, self.img_bpp, step_size)
                    # TODO: do something here
                frame_offset = file_info['nframes'] - (new_nframes // step_size) * step_size





