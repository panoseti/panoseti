import os
import shutil
import hashlib
import json

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

"""Directory name definitions"""
skycam_imgs_root_dir = 'skycam_imgs'
pano_imgs_root_dir = 'pano_imgs'
batch_data_root_dir = 'batch_data'
batch_labels_root_dir = 'batch_labels'
batch_data_zipfiles_dir = 'batch_data_zipfiles'

"""Json file name definitions"""
data_labels_fname = 'label_encoding.json'
feature_metadata_fname = 'feature_meta.json'


"""
Bata batch has the following file tree structure:
batch_data/
├─ task_cloud-detection.batch-id_0/
.
.
.
├─ task_cloud-detection.batch-id_N/
│  ├─ skycam_imgs/
│  │  ├─ SC2_imgs_2023-07-31/
│  │  │  ├─ original/
│  │  │  ├─ cropped/
│  │  │  ├─ pfov/
│  │  .
│  │  .
│  │  .
│  ├─ pano_imgs/
│  │  ├─ obs_Lick.start_2023-08-01T05:14:21Z.runtype_sci-obs.pffd/
│  │  │  ├─ original/
│  │  │  ├─ fft/
│  │  │  ├─ derivative/
│  │  .
│  │  .
│  │  .
│  ├─ skycam_path_index.json
│  ├─ pano_path_index.json
│  ├─ task_cloud-detection.batch-id_N.type_feature.csv
│  ├─ task_cloud-detection.batch-id_N.type_pano.csv
│  ├─ task_cloud-detection.batch-id_N.type_skycam.csv
.
.
.
"""


class CloudDetectionBatchDataFileTree:
    pano_path_index_fname = 'pano_path_index.json'
    skycam_path_index_fname = 'skycam_path_index.json'

    def __init__(self, batch_id, task='cloud-detection'):
        self.task = task
        self.batch_id = batch_id
        self.batch_path = get_batch_path(task, batch_id)
        self.batch_dir = get_batch_dir(task, batch_id)

        self.skycam_root_path = get_skycam_root_path(self.batch_path)
        self.pano_root_path = get_pano_root_path(self.batch_path)




class SkycamBatchDataFileTree(CloudDetectionBatchDataFileTree):
    def __init__(self, batch_id, **kwargs):
        super().__init__(batch_id)

        if 'skycam_dir' in kwargs:
            self.skycam_dir = kwargs['skycam_dir']
        elif all([key in kwargs for key in ['skycam_type', 'year', 'month', 'day']]):
            self.skycam_dir = self._get_skycam_dir(**kwargs)
        else:
            raise ValueError('Must provide skycam_dir or the info to create it.')
        self.skycam_path = f'{self.skycam_root_path}/{self.skycam_dir}'
        self.skycam_subdirs = self.get_skycam_subdirs()

    @staticmethod
    def _get_skycam_dir(skycam_type, year, month, day):
        if skycam_type == 'SC':
            return f'SC_imgs_{year}-{month:0>2}-{day:0>2}'
        elif skycam_type == 'SC2':
            return f'SC2_imgs_{year}-{month:0>2}-{day:0>2}'

    def get_skycam_subdirs(self):
        """Return dict of skycam image directories."""
        img_subdirs = {}
        for img_type in valid_skycam_img_types:
            img_subdirs[img_type] = f'{self.skycam_path}/{img_type}'
        return img_subdirs

    def get_skycam_img_path(self, original_fname, img_type):
        assert img_type in valid_skycam_img_types, f"{img_type} is not supported"
        if original_fname[-4:] != '.jpg':
            return None
        if img_type == 'original':
            return f"{self.skycam_subdirs['original']}/{original_fname}"
        elif img_type == 'cropped':
            return f"{self.skycam_subdirs['cropped']}/{original_fname[:-4]}_cropped.jpg"
        elif img_type == 'pfov':
            return f"{self.skycam_subdirs['pfov']}/{original_fname[:-4]}_pfov.jpg"
        else:
            return None


class PanoBatchDataFileTree(CloudDetectionBatchDataFileTree):
    def __init__(self, batch_id, run_dir):
        super().__init__(batch_id)

        self.pano_path = f'{self.pano_root_path}/{run_dir}'
        self.pano_subdirs = get_pano_subdirs(self.pano_path)


    def get_pano_img_path(self, pano_uid, img_type):
        assert img_type in valid_pano_img_types, f"{img_type} is not supported"
        return f"{self.pano_subdirs[img_type]}/pano-uid_{pano_uid}.feature-type_{img_type}.png"



"""Batch data directory"""

def get_batch_dir(task, batch_id):
    return "task_{0}.batch-id_{1}".format(task, batch_id)

def get_batch_path(task, batch_id):
    batch_path = batch_data_root_dir + '/' + get_batch_dir(task, batch_id)
    return batch_path

def get_root_dataset_dir(task):
    return f'dataset_{task}'

def get_user_label_export_dir(task, batch_id, user_uid, root):
    dir_name = "task_{0}.batch-id_{1}.user-uid_{2}".format(task, batch_id, user_uid)
    dir_path = f'{root}/{dir_name}'
    return dir_path

def get_batch_def_json_fname(task, batch_id):
    return f'name_batch-definition.task_{task}.batch-id_{batch_id}.json'



"""Skycam feature directory"""

valid_skycam_img_types = ['original', 'cropped', 'pfov']

def get_skycam_root_path(batch_path):
    skycam_imgs_root_path = f'{batch_path}/{skycam_imgs_root_dir}'
    return skycam_imgs_root_path

# def get_skycam_img_path(original_fname, img_type, skycam_path):
#     assert img_type in valid_skycam_img_types, f"{img_type} is not supported"
#     skycam_subdirs = get_skycam_subdirs(skycam_path)
#     if original_fname[-4:] != '.jpg':
#         return None
#     if img_type == 'original':
#         return f"{skycam_subdirs['original']}/{original_fname}"
#     elif img_type == 'cropped':
#         return f"{skycam_subdirs['cropped']}/{original_fname[:-4]}_cropped.jpg"
#     elif img_type == 'pfov':
#         return f"{skycam_subdirs['pfov']}/{original_fname[:-4]}_pfov.jpg"
#     else:
#         return None
#

# def get_skycam_path(batch_path, skycam_dir):
#     skycam_root_path = f'{batch_path}/{skycam_imgs_root_dir}'
#     skycam_path = f'{skycam_root_path}/{skycam_dir}'
#     return skycam_path
#


"""Pano feature directory"""


valid_pano_img_types = ['original', 'derivative', 'fft-derivative', 'fft']

def get_pano_subdirs(pano_path):
    pano_subdirs = {}
    for img_type in valid_pano_img_types:
        pano_subdirs[img_type] = f'{pano_path}/{img_type}'
    return pano_subdirs

def get_pano_root_path(batch_path):
    return f'{batch_path}/{pano_imgs_root_dir}'


def get_pano_img_path(pano_imgs_path, pano_uid, img_type):
    assert img_type in valid_pano_img_types, f"{img_type} is not supported"
    pano_subdirs = get_pano_subdirs(pano_imgs_path)
    return f"{pano_subdirs[img_type]}/pano-uid_{pano_uid}.feature-type_{img_type}.png"


"""UID definitions"""


def get_uid(data: str):
    """Returns SHA1 hash of a string input data."""
    data_bytes = data.encode('utf-8')
    data_uid = hashlib.sha1(data_bytes).hexdigest()
    return data_uid


def get_pano_uid(pano_original_fname, frame_offset):
    return get_uid(pano_original_fname + str(frame_offset))

def get_feature_uid(skycam_uid, pano_uid, batch_id):
    return get_uid(skycam_uid + pano_uid + str(batch_id))

def get_skycam_uid(original_skycam_fname):
    return get_uid(original_skycam_fname)


# Misc Utility


def parse_name(name):
    d = {}
    x = name.split('.')
    for s in x:
        y = s.split('_')
        if len(y) < 2:
            continue
        d[y[0]] = y[1]
    return d


def get_unix_from_datetime(t):
    return (t - datetime(1970, 1, 1, tzinfo=timezone.utc)) / timedelta(seconds=1)


def unpack_batch_data(batch_data_root_dir):
    """Unpack image files from batch data gztar file."""
    downloaded_fname = ''
    batch_dir = ''

    for fname in os.listdir(batch_data_root_dir):
        if fname.endswith('.tar.gz'):
            downloaded_fname = fname
            batch_dir = fname.rstrip('.tar.gz')
    if downloaded_fname:
        downloaded_fpath = f'{batch_data_root_dir}/{downloaded_fname}'
        batch_dir_path = f'{batch_data_root_dir}/{batch_dir}'
        print(f'Unzipping {downloaded_fpath}. This may take a minute...')
        shutil.unpack_archive(downloaded_fpath, batch_dir_path, 'gztar')
        os.remove(downloaded_fpath)
