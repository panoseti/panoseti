import os
import sys
from PIL import Image
import torchvision
import numpy as np

sys.path.append('../data_labeling')
import dataset_manager

class CloudDetection(torchvision.datasets.VisionDataset):

    def __init__(self, transform=None, target_transform=None):
        super(CloudDetection, self).__init__(None, transform=transform, target_transform=target_transform)
        assert os.path.exists("TEST_DATA.npy"), "You must upload the test data to the file system."
        self.data = [np.load("TEST_DATA.npy", allow_pickle=False)]

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        #self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index: int):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.data)
