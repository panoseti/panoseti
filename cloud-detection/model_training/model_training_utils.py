import os
import sys
from PIL import Image
import torchvision
import numpy as np

sys.path.append('../data_labeling')
# from ..data_labeling.dataset_manager import CloudDetectionDatasetManager
from dataset_manager import CloudDetectionDatasetManager

class CloudDetection(torchvision.datasets.VisionDataset):

    def __init__(self, transform=None, target_transform=None):
        super(CloudDetection, self).__init__(None, transform=transform, target_transform=target_transform)
        self.dataset_manager = CloudDetectionDatasetManager(root='../data_labeling')
        assert self.dataset_manager.verify_pano_feature_data(), "Not all pano feature data are valid."
        self.dsl_df = self.dataset_manager.main_dfs['dataset-labels']
        # self.data = [np.load("TEST_DATA.npy", allow_pickle=False)]
        #
        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # #self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index: int):
        feature_uid = self.dsl_df.loc[:, 'feature_uid'].iloc[index]
        pano_feature_fpath = self.dataset_manager.get_pano_feature_fpath(feature_uid, 'original')
        img = np.load(pano_feature_fpath, allow_pickle=False)
        # img = (img / 2**16)
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.dsl_df)

if __name__ == '__main__':
    training_data = CloudDetection(
        transform=None,  # NOTE: Make sure transform is the same as the one used in the training dataset.
    )
    training_data[0]
