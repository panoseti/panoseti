#! /usr/bin/env python3

import os
import json
from datetime import datetime, timedelta, timezone

from dataframe_utils import add_skycam_img, get_skycam_uid, skycam_imgs_root_dir, skycam_path_index_fname

valid_image_types = ['original', 'cropped', 'pfov']

def get_skycam_subdirs(skycam_path):
    """Return dict of skycam image directories."""
    img_subdirs = {}
    for img_type in valid_image_types:
        img_subdirs[img_type] = f'{skycam_path}/{img_type}'
    return img_subdirs


def get_skycam_img_path(original_fname, img_type, skycam_dir):
    assert img_type in valid_image_types, f"{img_type} is not supported"
    skycam_subdirs = get_skycam_subdirs(skycam_dir)
    if original_fname[-4:] != '.jpg':
        return None
    if img_type == 'original':
        return f"{skycam_subdirs['original']}/{original_fname}"
    elif img_type == 'cropped':
        return f"{skycam_subdirs['cropped']}/{original_fname[:-4]}_cropped.jpg"
    elif img_type == 'pfov':
        return f"{skycam_subdirs['pfov']}/{original_fname[:-4]}_pfov.jpg"
    else:
        return None


def get_skycam_dir(skycam_type, year, month, day):
    if skycam_type == 'SC':
        return f'SC_imgs_{year}-{month:0>2}-{day:0>2}'
    elif skycam_type == 'SC2':
        return f'SC2_imgs_{year}-{month:0>2}-{day:0>2}'

def get_skycam_root_path(batch_path):
    skycam_imgs_root_path = f'{batch_path}/{skycam_imgs_root_dir}'
    return skycam_imgs_root_path


def init_preprocessing_dirs(skycam_dir):
    """Initialize pre-processing directories."""
    img_subdirs = get_skycam_subdirs(skycam_dir)
    is_data_preprocessed(skycam_dir)
    #for dir_name in [img_subdirs['cropped'], img_subdirs['pfov']]:
    for dir_name in img_subdirs.values():
        os.makedirs(dir_name, exist_ok=True)


def is_initialized(skycam_dir):
    img_subdirs = get_skycam_subdirs(skycam_dir)
    if os.path.exists(skycam_dir) and len(os.listdir()) > 0:
        is_initialized = False
        for path in os.listdir():
            if path in img_subdirs:
                is_initialized |= len(os.listdir()) > 0
            if os.path.isfile(path):
                is_initialized = False
        if is_initialized:
            raise FileExistsError(f"Expected directory {skycam_dir} to be uninitialized, but found the following files:\n\t"
                                    f"{os.walk(skycam_dir)}")


def is_data_downloaded(skycam_dir):
    """Checks if data is already downloaded."""
    img_subdirs = get_skycam_subdirs(skycam_dir)
    if os.path.exists(img_subdirs['original']) and len(os.listdir(img_subdirs['original'])) > 0:
        raise FileExistsError(f"Data already downloaded at {img_subdirs['original']}")
    is_initialized(skycam_dir)

def is_data_preprocessed(skycam_dir):
    """Checks if data is already processed."""
    img_subdirs = get_skycam_subdirs(skycam_dir)
    if os.path.exists(img_subdirs['cropped']) and len(os.listdir(img_subdirs['cropped'])) > 0:
        raise FileExistsError(f"Data in {skycam_dir} already processed")
    is_initialized(skycam_dir)


def get_skycam_img_time(skycam_fname):
    """Returns datetime object based on the image timestamp contained in skycam_fname."""
    if skycam_fname[-4:] != '.jpg':
        raise Warning('Expected a .jpg file')
    # Example: SC2_20230625190102 -> SC2, 20230625190102
    skycam_type, t = skycam_fname[:-4].split('_')
    # 20230625190102 -> 2023, 06, 25, 19, 01, 02
    time_fields = t[0:4], t[4:6], t[6:8], t[8:10], t[10:12], t[12:14]
    year, month, day, hour, minute, second = [int(tf) for tf in time_fields]

    dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    tz = timezone(timedelta(hours=0))
    dt = dt.astimezone(tz)
    return dt


def get_batch_dir(task, batch_id):
    return "task_{0}.batch-id_{1}".format(task, batch_id)

def make_skycam_paths_json(batch_path):
    """Create file for indexing sky-camera image paths."""
    assert os.path.exists(batch_path), f"Could not find the batch directory {batch_path}"
    skycam_paths = {}
    skycam_imgs_root_path = get_skycam_root_path(batch_path)
    for path in os.listdir(skycam_imgs_root_path):
        skycam_path = f'{skycam_imgs_root_path}/{path}'
        if os.path.isdir(skycam_path) and 'SC' in path and 'imgs' in path:
            skycam_paths[skycam_path] = {
                "img_subdirs": {},
                "imgs_per_subdir": -1,
            }
            skycam_subdirs = get_skycam_subdirs(skycam_path)
            skycam_paths[skycam_path]["img_subdirs"] = skycam_subdirs
            num_imgs_per_subdir = []
            for subdir in skycam_subdirs.values():
                num_imgs_per_subdir.append(len(os.listdir(subdir)))
            if not all([e == num_imgs_per_subdir[0] for e in num_imgs_per_subdir]):
                raise Warning(f"Unequal number of images in {skycam_path}")
            skycam_paths[skycam_path]["imgs_per_subdir"] = num_imgs_per_subdir[0]
    with open(f"{batch_path}/{skycam_path_index_fname}", 'w') as f:
        f.write(json.dumps(skycam_paths, indent=4))
    return skycam_paths


def get_unix_from_datetime(t):
    return (t - datetime(1970, 1, 1, tzinfo=timezone.utc)) / timedelta(seconds=1)


def add_skycam_data_to_skycam_df(skycam_df, batch_id, skycam_imgs_root_path, skycam_dir, verbose):
    """Add entries for each skycam image to skycam_df """
    original_img_dir = get_skycam_subdirs(f'{skycam_imgs_root_path}/{skycam_dir}')['original']
    for original_skycam_fname in os.listdir(original_img_dir):
        if original_skycam_fname.endswith('.jpg'):
            # Collect image features
            skycam_type = original_skycam_fname.split('_')[0]
            t = get_skycam_img_time(original_skycam_fname)
            timestamp = get_unix_from_datetime(t)
            skycam_uid = get_skycam_uid(original_skycam_fname)
            # Add entries to skycam_df
            skycam_df = add_skycam_img(skycam_df, skycam_uid, original_skycam_fname, skycam_type, timestamp, batch_id, skycam_dir, verbose=verbose)
    return skycam_df


#make_skycam_paths_json('/Users/nico/panoseti/panoseti-software/cloud-detection/data_labeling/batch_data/task_cloud-detection.batch_0')
if __name__ == '__main__':
    make_skycam_paths_json('/Users/nico/panoseti/panoseti-software/cloud-detection/data_labeling/batch_data/task_cloud-detection.batch-id_0')