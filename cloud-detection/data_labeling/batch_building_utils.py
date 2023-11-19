import os
import shutil
import hashlib

from datetime import datetime, timedelta, timezone

skycam_imgs_root_dir = 'skycam_imgs'
pano_imgs_root_dir = 'pano_imgs'
batch_data_root_dir = 'batch_data'
batch_data_zipfiles_dir = 'batch_data_zipfiles'

data_labels_fname = 'label_encoding.json'
feature_metadata_fname = 'feature_meta.json'
pano_path_index_fname = 'pano_path_index.json'
skycam_path_index_fname = 'skycam_path_index.json'

def get_batch_dir(task, batch_id):
    return "task_{0}.batch-id_{1}".format(task, batch_id)

def get_batch_path(task, batch_id):
    batch_path = batch_data_root_dir + '/' + get_batch_dir(task, batch_id)
    return batch_path

def get_root_dataset_dir(task):
    return f'dataset_{task}'

def get_data_export_dir(task, batch_id, user_uid, root):
    dir_name = "task_{0}.batch-id_{1}.user-uid_{2}".format(task, batch_id, user_uid)
    dir_path = f'{root}/{dir_name}'
    return dir_path

def get_batch_def_json_fname(task, batch_id):
    return f'name_batch-definition.task_{task}.batch-id_{batch_id}.json'

# UID definitions

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
