import os
import numpy as np

from pano_utils import valid_pano_img_types, get_pano_subdirs
from batch_building_utils import get_root_dataset_dir, get_batch_dir

label_log_fname = 'label_log.json'
user_labeled_dir = 'user_labeled_batches'
batch_data_array_dir = 'batch_data_arrays'
pano_features_path = ''



# TODO: everything

# def get_batch_img_array_save_name(task, batch_id):
#     return f'name_batch-img-array.task_{task}.batch-id_{batch_id}.npy'
#
#

def write_dataset(data_dict):
    ...


def get_pano_dataset_feature_path(pano_dataset_path, pano_uid, img_type):
    assert img_type in valid_pano_img_types, f"{img_type} is not supported"
    pano_subdirs = get_pano_subdirs(pano_dataset_path)
    os.makedirs(pano_subdirs[img_type], exist_ok=True)
    return f"{pano_subdirs[img_type]}/pano-uid_{pano_uid}.feature-type_{img_type}.npy"

def get_pano_dataset_path(task, batch_id, run_dir):
    return f'{get_root_dataset_dir(task)}/{batch_data_array_dir}/{get_batch_dir(task, batch_id)}/{run_dir}'


def load_dataset():
    ...

def get_data(file_info_array, analysis_dir, step_size):
    # Save batch data to file
    npy_fname = gen_npy_fname(analysis_dir)
    if os.path.exists(npy_fname):
        data_arr = np.load(npy_fname)
        return data_arr



class PanoDatasetBuilder:
    supported_img_types = ['original', 'fft']#, 'derivatives']
    data_shapes = {
        'original': (32, 32),
        'fft': (32, 32),
        'derivatives': (3, 32, 32)
    }
    def __init__(self, task, batch_id, run_dir):
        self.pano_dataset_path = get_pano_dataset_path(task, batch_id, run_dir)
        os.makedirs(self.pano_dataset_path, exist_ok=True)
        self.task = task
        self.batch_id = batch_id
        self.pano_run_dir = run_dir
        self.data_arrays = dict()

    def clear_current_entry(self):
        self.data_arrays = dict()

    def add_img_to_entry(self, data, img_type):
        assert img_type in self.supported_img_types, f'img_type "{img_type}" not supported!'
        assert img_type not in self.data_arrays, f'img_type "{img_type}" already added!'
        self.data_arrays[img_type] = data

    def write_arrays(self, pano_uid, overwrite_ok=True):
        for img_type in self.data_arrays:
            fpath = get_pano_dataset_feature_path(self.pano_dataset_path, pano_uid, img_type)
            if os.path.exists(fpath) and not overwrite_ok:
                raise FileExistsError(f'overwrite_ok=False and {fpath} exists.')
            data = self.data_arrays[img_type]
            if data is None:
                raise ValueError(f'Data for "{img_type}" is None!')
            data = np.array(data)
            shape = self.data_shapes[img_type]
            if data.shape != shape:
                data = np.reshape(data, shape)
            np.save(fpath, data)
        self.clear_current_entry()


class PanoImgArrayBuilder:
    supported_img_types = ['original', 'fft', 'derivative-60s']
    def __init__(self, task, batch_id, run_dir):
        self.save_path = f'{get_root_dataset_dir(task)}/{batch_data_array_dir}/{get_batch_dir(task, batch_id)}/{run_dir}'
        os.makedirs(self.save_path, exist_ok=True)
        self.task = task
        self.batch_id = batch_id
        self.pano_run_dir = run_dir
        self.current_entry = dict()
        self.data_arrays = {
            img_type: [] for img_type in self.supported_img_types
        }

    def clear_current_entry(self):
        self.current_entry = dict()

    def commit_current_entry(self):
        entry = []
        for img_type in self.supported_img_types:
            assert img_type in self.current_entry, f'Missing {img_type} from the data array entry.'
            entry.append(self.current_entry[img_type])
            self.data_arrays[img_type].append(entry)
        self.clear_current_entry()

    def add_img_to_entry(self, data, img_type):
        assert img_type in self.supported_img_types, f'img_type "{img_type}" not supported!'
        assert img_type not in self.current_entry, f'img_type "{img_type}" already added!'
        self.current_entry[img_type] = data

    def write_array(self, overwrite_ok=True):
        fpath = f'{self.save_path}/{get_batch_img_array_save_name(self.task, self.batch_id)}'
        if os.path.exists(fpath) and not overwrite_ok:
            raise FileExistsError(f'overwrite_ok=False and {fpath} exists.')
        data = np.array(self.data_arrays)
        np.save(fpath, data)


