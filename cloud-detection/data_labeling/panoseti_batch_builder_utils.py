import os
import json
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../util")
import config_file
import pff
import image_quantiles

class ObservingRunFileInterface:

    def __init__(self, data_dir, run_dir):
        """File manager interface for a single observing run."""

        # Check data paths
        self.data_dir = data_dir
        self.run_dir = run_dir
        self.run_path = f'{self.data_dir}/{self.run_dir}'
        self.check_paths()

        self.data_config = config_file.get_data_config(self.run_path)
        self.obs_config = config_file.get_obs_config(self.run_path)

        # Create a dict of all valid pff files available for analysis, indexed by module.
        self.obs_pff_files = {}
        for dome in self.obs_config["domes"]:
            for module in dome["modules"]:
                module_id = config_file.ip_addr_to_module_id(module["ip_addr"])
                self.obs_pff_files[module_id] = self.get_module_file_info(module_id)

        # Unpack data_config attributes
        self.run_type = self.data_config["run_type"]
        self.has_imaging = False
        if "image" in self.data_config:
            self.has_imaging_data = True
            self.intgrn_usec = float(self.data_config["image"]["integration_time_usec"]) * 1e-6
            self.bpp = float(self.data_config["image"]["quabo_sample_size"]) // 2  # Bytes per imaging pixel
        self.has_pulse_height = False
        if "pulse_height" in self.data_config:
            self.has_pulse_height = True
            # TODO

    def check_paths(self):
        """Check if data_dir and run_dir exist."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"The data_directory '{self.data_dir}' does not exist!")
        elif not os.path.exists(self.run_path):
            raise FileNotFoundError(f"The run directory at '{self.run_path}' does not exist!")

    def get_module_file_info(self, module_id):
        """Returns an array of dictionaries storing info for all available pff files for module_id."""
        file_info_array = []
        for fname in os.listdir(self.run_path):
            fpath = f'{self.run_path}/{fname}'
            if not (pff.is_pff_file(fname)
                    and pff.pff_file_type(fname) in ['img16', 'img8']
                    and os.path.getsize(fpath) > 0):
                continue
            parsed_name = pff.parse_name(fname)
            if int(parsed_name["module"]) == module_id:
                attrs = {"fname": fname}
                with open(fpath, 'rb') as f:
                    bytes_per_pixel = int(parsed_name["bpp"])
                    img_size = bytes_per_pixel * 1024
                    attrs["frame_size"],attrs["nframes"], attrs["first_unix_t"], attrs["last_unix_t"] = pff.img_info(f, img_size)
                    attrs["img_size"] = img_size
                    attrs["bytes_per_pixel"] = bytes_per_pixel
                    attrs["module"] = int(parsed_name["module"])
                    attrs["seqno"] = int(parsed_name["seqno"])
                file_info_array += [attrs]
        # Sort files_to_process in ascending order by file sequence number
        file_info_array.sort(key=lambda attrs: attrs["seqno"])
        return file_info_array




class ModuleImageInterface(ObservingRunFileInterface):

    def __init__(self, data_dir, run_dir, module_id):
        super().__init__(data_dir, run_dir)
        if module_id not in self.obs_pff_files:
            raise ValueError(f"Module {module_id} does not exist in the run '{run_dir}'.")
        self.module_id = module_id

        # Process file info
        self.module_pff_files = self.module_pff_files[module_id]

        if len(self.module_pff_files) == 0:
            raise Warning(f"No valid panoseti imaging data found at '{self.run_path}'")
        self.start_unix_t = self.module_pff_files[0]['first_unix_t']
        print(self.module_pff_files)

    def get_next_frame(self, fp, frame_size, bytes_per_pixel, step_size):
        """Returns the next image frame and json header from the panoseti file pointer fp."""
        j = json.loads(pff.read_json(fp))
        img = pff.read_image(fp, 32, bytes_per_pixel)
        fp.seek((step_size - 1) * frame_size, os.SEEK_CUR)  # Skip (step_size - 1) images
        return img, j


class PanosetiBatchBuilder(ObservingRunFileInterface):
    def __init__(self, data_dir, run_dir, task, batch_id):
        super().__init__(data_dir, run_dir)
        self.task = task
        self.batch_id = batch_id

        self.batch_dir = ...

    def get_panoseti_batch_dir(self):
        ...

    def gen_npy_fname(self):
        npy_fname = f"{self.batch_dir}/data.npy"
        return npy_fname


    def get_empty_data_array(self, file_attrs, step_size):
        data_size = 0
        for i in range(len(file_attrs)):
            data_size += file_attrs[i]['nframes'] // step_size
        return np.zeros(data_size)

    def get_files_to_process(self, data_dir, run_dir, module):
        files_to_process = []
        for fname in os.listdir(f'{data_dir}/{run_dir}'):
            if pff.is_pff_file(fname) and pff.pff_file_type(fname) in ('img16', 'img8'):
                files_to_process.append(fname)
        return files_to_process

    def process_file(self, file_info, data, itr_info, step_size):
        """On a sample of the frames in the file represented by file_info, add the total
        image brightness to the data array beginning at data_offset."""
        with open(f"{self.data_dir}/{self.run_dir}/{file_info['fname']}", "rb") as f:
            # Start file pointer with an offset based on the previous file -> ensures even frame sampling
            f.seek(
                itr_info['fstart_offset'] * file_info['frame_size'],
                os.SEEK_CUR
            )
            new_nframes = file_info['nframes'] - itr_info['fstart_offset']
            for i in range(new_nframes // step_size):
                img, j = self.get_next_frame(f,
                                             file_info['frame_size'],
                                             file_info['bytes_per_pixel'],
                                             step_size)
                data[itr_info['data_offset'] + i] = np.sum(img)
            itr_info['fstart_offset'] = file_info['nframes'] - (new_nframes // step_size) * step_size


    def get_data(self, file_info_array, analysis_dir, step_size):
        # Save reduced data to file
        npy_fname = self.gen_npy_fname(analysis_dir)
        if os.path.exists(npy_fname):
            data_arr = np.load(npy_fname)
            return data_arr
        itr_info = {
            "data_offset": 0,
            "fstart_offset": 0  # Ensures frame step size across files
        }
        data_arr = self.get_empty_data_array(file_info_array, step_size)
        for i in range(len(file_info_array)):
            print(f"Processing {file_info_array[i]['fname']}")
            file_info = file_info_array[i]
            self.process_file(file_info, data_arr, itr_info, step_size)
            itr_info['data_offset'] += file_info["nframes"] // step_size

        np.save(npy_fname, data_arr)
        return data_arr


if __name__ == '__main__':
    DATA_DIR = '/Users/nico/Downloads/panoseti_test_data/obs_data/data'
    RUN_DIR = 'obs_Lick.start_2023-08-01T05:14:21Z.runtype_sci-obs.pffd'

    test_batch_builder = PanosetiBatchBuilder(DATA_DIR, RUN_DIR, 'cloud-detection', 0)
    test_module_interface = ModuleImageInterface(DATA_DIR, RUN_DIR, 4)
