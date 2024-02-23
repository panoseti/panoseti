#! /usr/bin/env python3
import json
from batch_builder import CloudDetectionBatchBuilder
from batch_building_utils import load_batch_def

# Build features for a prediction dataset
batch_id = 11
batch_def = load_batch_def(batch_id, 'prediction')['batch-def']
print(batch_def)

data_batch_builder = CloudDetectionBatchBuilder(
    batch_id,
    batch_def,
    batch_type='inference',
    force_recreate=True
)
data_batch_builder.build_inference_batch()
