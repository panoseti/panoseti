
import torch
import torchvision
from torchvision.transforms import v2
from torch import nn
from torchsummary import summary

from training_utils import *
from data_loaders import *

class CloudDetection(nn.Module):
    input_shape = (2, 32, 32)

    def __init__(self):
        super().__init__()
        conv1_groups = 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 126, 3, stride=1, padding='same', groups=conv1_groups),
            nn.ReLU(),
            nn.BatchNorm2d(126),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(126, 126, 3, stride=1, padding='same', groups=conv1_groups),
            nn.ReLU(),
            nn.BatchNorm2d(126),
            nn.Dropout2d(p=0.5),

            # nn.Conv2d(126, 126, 3, stride=1, padding='same', groups=conv1_groups),
            # nn.ReLU(),
            # nn.BatchNorm2d(126),
            # nn.Dropout2d(p=0.5),

            nn.MaxPool2d(kernel_size=3),
        )
        conv2_groups = 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(126, 200, 3, stride=1, padding='same', groups=conv2_groups),
            nn.ReLU(),
            nn.BatchNorm2d(200),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(200, 200, 3, stride=1, padding='same', groups=conv2_groups),
            nn.ReLU(),
            nn.BatchNorm2d(200),
            nn.Dropout2d(p=0.5),

            # nn.Conv2d(200, 200, 3, stride=1, padding='same', groups=conv2_groups),
            # nn.ReLU(),
            # nn.BatchNorm2d(200),
            # nn.Dropout2d(p=0.5),

            nn.MaxPool2d(kernel_size=3),
        )
        
        # conv3_groups = 1
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(200, 256, 3, stride=1, padding='same', groups=conv3_groups),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(p=0.5),
        
        #     nn.Conv2d(256, 256, 3, stride=1, padding='same', groups=conv3_groups),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(p=0.5),
        
        #     nn.Conv2d(256, 256, 3, stride=1, padding='same', groups=conv3_groups),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(p=0.5),
        
        #     nn.MaxPool2d(kernel_size=3),
        # )
        
        # conv4_groups = 1
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(256, 512, 3, stride=1, padding='same', groups=conv4_groups),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(512),
        #     nn.Dropout2d(p=0.5),
        
        #     nn.Conv2d(512, 512, 3, stride=1, padding='same', groups=conv4_groups),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(512),
        #     nn.Dropout2d(p=0.5),
        
        #     nn.Conv2d(512, 512, 3, stride=1, groups=conv4_groups),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(512),
        #     nn.Dropout2d(p=0.5),
        
        #     nn.MaxPool2d(kernel_size=2)
        # )

        self.flatten = nn.Flatten()

        self.linear_stack = nn.Sequential(
            nn.LazyLinear(500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Dropout1d(p=0.5),

            nn.LazyLinear(500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Dropout1d(p=0.5),

            # nn.LazyLinear(512),
            # nn.ReLU(),
            # nn.BatchNorm1d(512),
            # nn.Dropout1d(p=0.5),

            # nn.LazyLinear(256),
            # nn.ReLU(),
            # nn.BatchNorm1d(256),

            nn.LazyLinear(2),
        )

        # self.cnns = torch.nn.ModuleList([torch.nn.Sequential(self.conv1(), self.conv2(), self.conv3(), self.flatten()) for _ in range(3)])

    # def forward_convolve(self, x):
    #     out = self.conv1(x)
    #     out = self.conv2(out)
    #     out = self.conv3(out)
    #     out = self.conv4(out)
    #     out = self.flatten(out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.conv3(out)
        # out = self.conv4(out)
        out = self.flatten(out)
        # print(out.shape)

        out = self.linear_stack(out)
        return out
