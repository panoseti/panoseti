import os
import sys
from PIL import Image

import torch
from torch import nn
from torchsummary import summary
import torchvision

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import tqdm.notebook as tqdm

sys.path.append('../dataset_construction')
# from ..data_labeling.dataset_manager import CloudDetectionDatasetManager
from dataset_builder import CloudDetectionDatasetManager
from dataset_utils import PanoDatasetBuilder
from batch_building_utils import valid_pano_img_types

class CloudDetectionTrain(torchvision.datasets.VisionDataset):

    def __init__(self, transform=None, target_transform=None):
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
            data = np.load(pano_feature_fpath, allow_pickle=False)
            if self.transform is not None:
                data = self.transform(data)
            img_data[img_type] = data
        return img_data, y

    def __len__(self) -> int:
        return len(self.dsl_df)


class CloudDetectionPredict(torchvision.datasets.VisionDataset):
    # TODO: replace labeled dataset with dataset with no labels
    pass

    def __init__(self, transform=None, target_transform=None):
        super().__init__(None, transform=transform, target_transform=target_transform)
        self.dataset_manager = CloudDetectionDatasetManager(batch_type='prediction', root='../dataset_construction')
        assert self.dataset_manager.verify_pano_feature_data(), "Not all pano feature data are valid."
        self.dsl_df = self.dataset_manager.main_dfs['dataset-labels']
        self.one_hot_encoding = self.dataset_manager.get_one_hot_encoding()

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
            data = np.load(pano_feature_fpath, allow_pickle=False)
            if self.transform is not None:
                data = self.transform(data)
            img_data[img_type] = data
        return img_data, y

    def __len__(self) -> int:
        return len(self.dsl_df)


