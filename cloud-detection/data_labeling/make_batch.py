#! /usr/bin/env python3
from dataset_builder import *
from batch_builder import CloudDetectionBatchBuilder

DATA_DIR = '/Users/nico/Downloads/panoseti_test_data/obs_data/data'

batch_def_3 = [
    {
        'pano': {
            'data_dir': DATA_DIR,
            'run_dir': 'obs_Lick.start_2023-08-01T05:14:21Z.runtype_sci-obs.pffd',
        },
        'skycam': {
            'skycam_type': 'SC2',
            'year': 2023,
            'month': 7,
            'day': 31
        }
    },
    # {
    #     'pano': {
    #         'data_dir': DATA_DIR,
    #         'run_dir': 'obs_Lick.start_2023-08-24T04:37:00Z.runtype_sci-obs.pffd',
    #     },
    #     'skycam': {
    #         'skycam_type': 'SC2',
    #         'year': 2023,
    #         'month': 8,
    #         'day': 23
    #     }
    # },
    {
        'pano': {
            'data_dir': DATA_DIR,
            'run_dir': 'obs_Lick.start_2023-08-29T04:49:58Z.runtype_sci-obs.pffd',
        },
        'skycam': {
            'skycam_type': 'SC2',
            'year': 2023,
            'month': 8,
            'day': 28
        }
    },
]
batch_id = 6

data_batch_builder = CloudDetectionBatchBuilder(
    batch_id,
    batch_def_3,
    force_recreate=True
)
data_batch_builder.build_batch()

