import os
import sys

import torch
from torchvision.transforms import v2

import torchvision

import numpy as np

sys.path.append('../dataset_construction')
from dataset_builder import CloudDetectionDatasetManager
from inference_session import InferenceSession

default_transform = v2.Compose([
    # v2.RandomResizedCrop(size=(224, 224), antialias=True),
    # v2.RandomHorizontalFlip(p=0.5),
    v2.ToTensor(),
    v2.ToDtype(torch.float64, scale=False)
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CloudDetectionTrain(torchvision.datasets.VisionDataset):

    def __init__(self, transform=default_transform, target_transform=None):
        super().__init__(None, transform=transform, target_transform=target_transform)
        self.dataset_manager = CloudDetectionDatasetManager(batch_type='training', root='../dataset_construction')
        assert self.dataset_manager.verify_pano_feature_data(), "Not all pano feature data are valid."
        self.dsl_df = self.dataset_manager.main_dfs['dataset-labels']
        self.one_hot_encoding = self.dataset_manager.get_one_hot_encoding()
        # if pano_img_type not in [t for t in valid_pano_img_types if 'raw' in t]:
        #     raise ValueError(f'"{pano_img_type} is not a supported feature type')
        # self.pano_img_type = pano_img_type

    def __getitem__(self, index: int):
        feature_uid, label = self.dsl_df.loc[:, ['feature_uid', 'label']].iloc[index]
        y = self.one_hot_encoding[label]

        img_data = {
            'raw-original': None,
            'raw-fft': None,
            'raw-derivative.-60': None
        }
        for img_type in img_data:
            pano_feature_fpath = self.dataset_manager.get_pano_feature_fpath(feature_uid, img_type)
            data = np.load(pano_feature_fpath, allow_pickle=False).astype(np.double) / 2**10
            if self.transform is not None:
                data = self.transform(data)
            img_data[img_type] = data
        return img_data, y

    def __len__(self) -> int:
        return len(self.dsl_df)


class CloudDetectionInference(torchvision.datasets.VisionDataset):
    def __init__(self, batch_id, transform=default_transform, target_transform=None):
        super().__init__(None, transform=transform, target_transform=target_transform)
        self.inference_session = InferenceSession(batch_id)

    def __getitem__(self, index: int):
        feature_uid = self.inference_session.unlabeled_df.loc[:, 'feature_uid'].iloc[index]
        img_data = {
            'raw-original': None,
            'raw-fft': None,
            'raw-derivative.-60': None
        }
        for img_type in img_data:
            pano_feature_fpath = self.inference_session.get_pano_feature_fpath(feature_uid, img_type)
            data = np.load(pano_feature_fpath, allow_pickle=False).astype(np.double) / 2**10
            if self.transform is not None:
                data = self.transform(data)
            img_data[img_type] = data
        return img_data

    def __len__(self) -> int:
        return len(self.inference_session.unlabeled_df)


if __name__ == '__main__':
    os.chdir('../model_training')
    inference_data = CloudDetectionInference(
        batch_id=10,
    )
    inference_loader = torch.utils.data.DataLoader(
        dataset=inference_data,
    )