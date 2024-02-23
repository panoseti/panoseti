#! /usr/bin/env python3
import json
from batch_builder import CloudDetectionBatchBuilder
from batch_building_utils import load_batch_def

# DATA_DIR = '/Users/nico/Downloads/panoseti_test_data/obs_data/data'

# Build features for a training dataset
batch_id = 10
batch_def = load_batch_def(batch_id, 'training')['batch-def']
print(batch_def)

data_batch_builder = CloudDetectionBatchBuilder(
    batch_id,
    batch_def,
    batch_type='training',
    force_recreate=True
)
data_batch_builder.build_batch()

print('done')
