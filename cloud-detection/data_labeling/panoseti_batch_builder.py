
import os
import numpy as np

import matplotlib.pyplot as plt

from panoseti_file_interfaces import ObservingRunFileInterface, ModuleImageInterface
from skycam_utils import get_batch_dir
from panoseti_batch_utils import *
from dataframe_utils import *


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

    def process_file(self, file_info, data, itr_info, step_size):
        """On a sample of the frames in the file represented by file_info, add the total
        image brightness to the data array beginning at data_offset."""
        with open(f"{self.data_dir}/{self.run_dir}/{file_info['fname']}", 'rb') as f:
            # Start file pointer with an offset based on the previous file -> ensures even frame sampling
            f.seek(
                itr_info['fstart_offset'] * file_info['frame_size'],
                os.SEEK_CUR
            )
            new_nframes = file_info['nframes'] - itr_info['fstart_offset']
            for i in range(new_nframes // step_size):
                j, img = self.get_next_frame(f, file_info['frame_size'], file_info['bytes_per_pixel'], step_size)
                data[itr_info['data_offset'] + i] = np.sum(img)
            itr_info['fstart_offset'] = file_info['nframes'] - (new_nframes // step_size) * step_size




