#! /usr/bin/env python3

"""
Routines for building data batch skycam features.
"""
import os
import json

from batch_building_utils import *


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
                if subdir.endswith('original'):
                    continue
                num_imgs_per_subdir.append(len(os.listdir(subdir)))
            if not all([e == num_imgs_per_subdir[0] for e in num_imgs_per_subdir]):
                raise Warning(f"Unequal number of images in {skycam_path}")
            skycam_paths[skycam_path]["imgs_per_subdir"] = num_imgs_per_subdir[0]
    with open(f"{batch_path}/{skycam_path_index_fname}", 'w') as f:
        f.write(json.dumps(skycam_paths, indent=4))
    return skycam_paths



#make_skycam_paths_json('/Users/nico/panoseti/panoseti-software/cloud-detection/data_labeling/batch_data/task_cloud-detection.batch_0')
if __name__ == '__main__':
    make_skycam_paths_json(
        '/panoseti-software/cloud-detection/data_labeling/batch_data1/task_cloud-detection.batch-id_0')