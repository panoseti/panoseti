#! /usr/bin/env python3
import json
from batch_builder import CloudDetectionBatchBuilder
from batch_building_utils import batch_defs_fname

DATA_DIR = '/Users/nico/Downloads/panoseti_test_data/obs_data/data'


def load_batch_defs():
    with open(batch_defs_fname, 'r') as f:
        return json.load(f)

def load_batch_def(batch_id):
    batch_defs = load_batch_defs()
    batch_def = [batch for batch in batch_defs['batches'] if batch['batch-id'] == batch_id]
    if len(batch_def) == 0:
        raise ValueError(f'No batch definitions exist for batch_id={batch_id}.')
    elif len(batch_def) > 1:
        raise ValueError(f'Found multiple batch definitions with batch_id={batch_id}:\n'
                         f'{batch_def}')
    else:
        return batch_def[0]


batch_id = 1
batch_def = load_batch_def(batch_id)['batch-def']
print(batch_def)

data_batch_builder = CloudDetectionBatchBuilder(
    batch_id,
    batch_def,
    force_recreate=True
)
data_batch_builder.build_batch()

