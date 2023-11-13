
import json
import os
import numpy as np
import matplotlib.pyplot as plt


from dataframe_utils import add_pano_img, pano_imgs_root_dir, pano_path_index_fname


valid_image_types = ['original', 'derivative', 'fft-derivative', 'fft']

# File structure abstraction
def get_pano_subdirs(pano_path):
    pano_subdirs = {}
    for img_type in valid_image_types:
        pano_subdirs[img_type] = f'{pano_path}/{img_type}'
    return pano_subdirs

def get_pano_root_path(batch_path):
    return f'{batch_path}/{pano_imgs_root_dir}'

def get_pano_img_path(pano_imgs_path, pano_uid, img_type):
    assert img_type in valid_image_types, f"{img_type} is not supported"
    pano_subdirs = get_pano_subdirs(pano_imgs_path)
    return f"{pano_subdirs[img_type]}/pano-uid_{pano_uid}.feature-type_{img_type}.png"
    # if feature_type == 'original':
    #     return f"{pano_subdirs['original']}/{original_fname.rstrip('.pff')}.feature-type_original.pff"
    # elif feature_type == 'derivative':
    #     return f"{pano_subdirs['derivative']}/{original_fname.rstrip('.pff')}.feature-type_derivative.pff"
    # elif feature_type == 'fft':
    #     return f"{pano_subdirs['fft']}/{original_fname.rstrip('.pff')}.feature-type_.pff"
    # else:
    #     return None
    #


def make_pano_paths_json(batch_path):
    """Create file for indexing sky-camera image paths."""
    assert os.path.exists(batch_path), f"Could not find the batch directory {batch_path}"
    pano_paths = {}
    pano_imgs_root_path = get_pano_root_path(batch_path)
    for path in os.listdir(pano_imgs_root_path):
        pano_path = f'{pano_imgs_root_path}/{path}'
        if os.path.isdir(pano_path) and 'pffd' in path:
            pano_paths[pano_path] = {
                "img_subdirs": {},
                "imgs_per_subdir": -1,
            }
            pano_subdirs = get_pano_subdirs(pano_path)
            pano_paths[pano_path]["img_subdirs"] = pano_subdirs
            num_imgs_per_subdir = []
            for subdir in pano_subdirs.values():
                num_imgs_per_subdir.append(len(os.listdir(subdir)))
            if not all([e == num_imgs_per_subdir[0] for e in num_imgs_per_subdir]):
                raise Warning(f"Unequal number of images in {pano_path}")
            pano_paths[pano_path]["imgs_per_subdir"] = num_imgs_per_subdir[0]
    with open(f"{batch_path}/{pano_path_index_fname}", 'w') as f:
        f.write(json.dumps(pano_paths, indent=4))
    return pano_paths



def add_pano_data_to_pano_df(pano_df, batch_id, pano_imgs_root_path, pano_dir, verbose):
    """Add entries for each skycam image to skycam_df """
    original_img_dir = get_pano_subdirs(f'{pano_imgs_root_path}/{pano_dir}')['original']
    for original_fname in os.listdir(original_img_dir):
        if original_fname[-4:] == '.jpg':
            # Collect image features

            # Add entries to skycam_df
            pano_df = add_pano_img(pano_df, ...)
    return pano_df



if __name__ == '__main__':
    make_pano_paths_json('/Users/nico/panoseti/panoseti-software/cloud-detection/data_labeling/batch_data/task_cloud-detection.batch-id_0')
