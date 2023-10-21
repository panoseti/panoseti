#! /usr/bin/env python3
import os
import random

from skycam_utils import get_batch_dir
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

def init_batch(task, batch_id):
    batch_path = make_batch_dir(task, batch_id)
    for sample in samples:
        preprocess_skycam_imgs(sample['skycam_type'],
                               sample['year'],
                               sample['month'],
                               sample['day'],
                               root=batch_path)
        

if __name__ == '__main__':
    init_batch(task='cloud-detection', batch_id=1)