
import torch
import torchvision
from torchvision.transforms import v2
from torch import nn
from torchsummary import summary

from training_utils import *
from data_loaders import *

class CloudDetection(nn.Module):
    input_shape = (3, 32, 32)

    def __init__(self):
        super().__init__()
        
        conv1_groups = 3
        conv1_nker = 27
        conv1_kernel_size = 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_nker, conv1_kernel_size, stride=1, padding='same', groups=conv1_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv1_nker),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(conv1_nker, conv1_nker, conv1_kernel_size, stride=1, padding='same', groups=conv1_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv1_nker),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(conv1_nker, conv1_nker, conv1_kernel_size, stride=1, padding='same', groups=conv1_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv1_nker),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(p=0.5)
        )

        conv2_groups = 1
        conv2_nker = 27
        conv2_kernel_size = 5
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_nker, conv2_nker, conv2_kernel_size, stride=1, padding='same', groups=conv2_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv2_nker),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(conv2_nker, conv2_nker, conv2_kernel_size, stride=1, padding='same', groups=conv2_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv2_nker),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(p=0.5),
        )
        
        # conv3_groups = 1
        # conv3_nker = 32
        # conv3_kernel_size = 3
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(conv2_nker, conv3_nker, conv3_kernel_size, stride=1, padding='same', groups=conv3_groups),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(conv3_nker),
        #     nn.Dropout2d(p=0.5),
        #
        #     nn.Conv2d(conv3_nker, conv3_nker, conv3_kernel_size, stride=1, padding='same', groups=conv3_groups),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(conv3_nker),
        #     nn.MaxPool2d(kernel_size=3),
        #     nn.Dropout2d(p=0.5),
        #
        # )

        self.flatten = nn.Flatten()

        self.linear_stack = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout1d(p=0.5),

            nn.LazyLinear(84),
            nn.ReLU(),
            nn.BatchNorm1d(84),
            nn.Dropout1d(p=0.5),
            
            nn.LazyLinear(2),
        )


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.conv3(out)
        out = self.flatten(out)

        out = self.linear_stack(out)
        return out
