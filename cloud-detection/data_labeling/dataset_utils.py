import os
import numpy as np


# TODO: everything
def gen_npy_fname(analysis_dir):
    npy_fname = f"{analysis_dir}/data.npy"
    return npy_fname


def write_dataset(data_dict):
    ...


def load_dataset():
    ...

def get_data(file_info_array, analysis_dir, step_size):
    # Save batch data to file
    npy_fname = gen_npy_fname(analysis_dir)
    if os.path.exists(npy_fname):
        data_arr = np.load(npy_fname)
        return data_arr

