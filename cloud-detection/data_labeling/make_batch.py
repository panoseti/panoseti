#! /usr/bin/env python3
import os
import random
import shutil

from skycam_utils import get_batch_dir, make_skycam_paths_json
from preprocess_skycam import preprocess_skycam_imgs

def make_batch_dir(task, batch_id, root='batch_data'):
    batch_path = root + '/' + get_batch_dir(task, batch_id)
    os.makedirs(batch_path, exist_ok=True)
    return batch_path


def select_samples():
    ...

samples = [
    {
        'skycam_type': 'SC2',
        'year': 2023,
        'month': 7,
        'day': 24
    },
    {
        'skycam_type': 'SC2',
        'year': 2023,
        'month': 6,
        'day': 24
    },
    {
        'skycam_type': 'SC2',
        'year': 2020,
        'month': 6,
        'day': 24
    },
    {
        'skycam_type': 'SC2',
        'year': 2021,
        'month': 6,
        'day': 24
    },
    {
        'skycam_type': 'SC2',
        'year': 2023,
        'month': 6,
        'day': 24
    }
]


def init_batch(task, batch_id, do_zip=False):
    batch_path = make_batch_dir(task, batch_id)
    for sample in samples:
        preprocess_skycam_imgs(sample['skycam_type'],
                               sample['year'],
                               sample['month'],
                               sample['day'],
                               root=batch_path)
    make_skycam_paths_json(batch_path)
    if do_zip:
        zip_batch(task, batch_id)


def zip_batch(task, batch_id, root='batch_data_zipfiles'):
    batch_path = batch_path = make_batch_dir(task, batch_id)
    batch_dir = get_batch_dir(task, batch_id)
    batch_zip_name = f'{root}/{batch_dir}'
    if not os.path.exists(batch_zip_name + '.tar.gz'):
        print(f"Zipping {batch_dir}")
        os.makedirs(root, exist_ok=True)
        shutil.make_archive(batch_zip_name, 'gztar', batch_path)


if __name__ == '__main__':
    init_batch(task='cloud-detection', batch_id=1, do_zip=True)
