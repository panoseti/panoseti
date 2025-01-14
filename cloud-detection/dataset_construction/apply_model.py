#! /usr/bin/env python3
import json
from batch_builder import CloudDetectionBatchBuilder
from batch_building_utils import load_batch_def

def build_features(batch_id):
    # Build features for a prediction dataset
    batch_def = load_batch_def(batch_id, 'inference')['batch-def']
    print(f'Building:\n{batch_def}')

    data_batch_builder = CloudDetectionBatchBuilder(
        batch_id,
        batch_def,
        batch_type='inference',
        force_recreate=True
    )
    data_batch_builder.build_inference_batch()

batch_ids = list(range(1000, 1011))
#batch_ids = [1011]#list(range(0, 11))
for batch_id in batch_ids:
    build_features(batch_id)

# inference_data = CloudDetectionInference(
#     batch_id = 10,
#     transform = transform
# )
# inference_loader = torch.utils.data.DataLoader(
#   dataset=inference_data,
#   batch_size=batch_size
# )
#

