import os
import sys
from PIL import Image
import torchvision
import numpy as np

sys.path.append('../data_labeling')
# from ..data_labeling.dataset_manager import CloudDetectionDatasetManager
from dataset_builder import CloudDetectionDatasetManager
from dataset_utils import PanoDatasetBuilder
from batch_building_utils import valid_pano_img_types

class CloudDetectionTrain(torchvision.datasets.VisionDataset):

    def __init__(self, transform=None, target_transform=None, pano_img_type='raw-original'):
        super().__init__(None, transform=transform, target_transform=target_transform)
        self.dataset_manager = CloudDetectionDatasetManager(root='../data_labeling')
        assert self.dataset_manager.verify_pano_feature_data(), "Not all pano feature data are valid."
        self.dsl_df = self.dataset_manager.main_dfs['dataset-labels']
        self.one_hot_encoding = self.dataset_manager.get_one_hot_encoding()
        if pano_img_type not in [t for t in valid_pano_img_types if 'raw' in t]:
            raise ValueError(f'"{pano_img_type} is not a supported feature type')
        self.pano_img_type = pano_img_type

    def __getitem__(self, index: int):
        feature_uid, label = self.dsl_df.loc[:, ['feature_uid', 'label']].iloc[index]
        pano_feature_fpath = self.dataset_manager.get_pano_feature_fpath(feature_uid, self.pano_img_type)
        img = np.load(pano_feature_fpath, allow_pickle=False)
        y = self.one_hot_encoding[label]
        if self.transform is not None:
            img = self.transform(img)
        return img, y

    def __len__(self) -> int:
        return len(self.dsl_df)


class CloudDetectionPredict(torchvision.datasets.VisionDataset):
    # TODO: replace labeled dataset with dataset with no labels
    pass