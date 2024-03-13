
import torch
import torchvision
from torchvision.transforms import v2
from torch import nn
from torchsummary import summary

from training_utils import *
from data_loaders import *

class CloudDetection(nn.Module):
    input_shape = (1, 32, 32)

    def __init__(self):
        super().__init__()
        
        conv1_groups = 1
        conv1_nker = 24
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv1_nker, 3, stride=1, padding='same', groups=conv1_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv1_nker),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(conv1_nker, conv1_nker, 3, stride=1, padding='same', groups=conv1_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv1_nker),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(kernel_size=3),
            

            # nn.Conv2d(conv1_nker, conv1_nker, 3, stride=1, padding='same', groups=conv1_groups),
            # nn.ReLU(),
            # nn.BatchNorm2d(conv1_nker),
            # nn.Dropout2d(p=0.5),
            # nn.MaxPool2d(kernel_size=2),
            
        )
        
        conv2_groups = 1
        conv2_nker = 48
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_nker, conv2_nker, 3, stride=1, padding='same', groups=conv2_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv2_nker),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(conv2_nker, conv2_nker, 3, stride=1, padding='same', groups=conv2_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv2_nker),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(kernel_size=3),

            # nn.MaxPool2d(kernel_size=2),
            
            # nn.Conv2d(conv2_nker, conv2_nker, 3, stride=1, padding='same', groups=conv2_groups),
            # nn.ReLU(),
            # nn.BatchNorm2d(conv2_nker),
            # nn.Dropout2d(p=0.5),
            # nn.MaxPool2d(kernel_size=2),

            # nn.Conv2d(conv2_nker, conv2_nker, 3, stride=1, padding='same', groups=conv2_groups),
            # nn.ReLU(),
            # nn.BatchNorm2d(conv2_nker),
            # nn.Dropout2d(p=0.5),
            # # nn.MaxPool2d(kernel_size=2),

            # nn.Conv2d(conv2_nker, conv2_nker, 3, stride=1, padding='same', groups=conv2_groups),
            # nn.ReLU(),
            # nn.BatchNorm2d(conv2_nker),
            # nn.Dropout2d(p=0.5),
            # nn.MaxPool2d(kernel_size=2),


            # nn.Conv2d(conv2_nker, conv2_nker, 3, stride=1, padding='same', groups=conv2_groups),
            # nn.ReLU(),
            # nn.BatchNorm2d(conv2_nker),
            # nn.Dropout2d(p=0.5),
            # nn.MaxPool2d(kernel_size=2),
        )
        
        # conv3_groups = 1
        # conv3_nker = 128
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(conv2_nker, conv3_nker, 3, stride=1, padding='same', groups=conv3_groups),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(conv3_nker),
        #     nn.Dropout2d(p=0.5),
        
        #     nn.Conv2d(conv3_nker, conv3_nker, 3, stride=1, padding='same', groups=conv3_groups),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(conv3_nker),
        #     nn.Dropout2d(p=0.5),
        #     nn.MaxPool2d(kernel_size=2),

        
        #     # nn.Conv2d(conv3_nker, conv3_nker, 3, stride=1, padding='same', groups=conv3_groups),
        #     # nn.ReLU(),
        #     # nn.BatchNorm2d(conv3_nker),
        #     # nn.Dropout2d(p=0.5),
        #     # nn.MaxPool2d(kernel_size=2),
        # )
        
        # conv4_groups = 1
        # conv4_nker = 50
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(conv3_nker, conv4_nker, 3, stride=1, padding='same', groups=conv4_groups),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(conv4_nker),
        #     nn.Dropout2d(p=0.5),
        
        #     nn.Conv2d(conv4_nker, conv4_nker, 3, stride=1, padding='same', groups=conv4_groups),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(conv4_nker),
        #     nn.Dropout2d(p=0.5),
        
        #     nn.Conv2d(conv4_nker, conv4_nker, 3, stride=1, groups=conv4_groups),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(conv4_nker),
        #     nn.Dropout2d(p=0.5),
        
        #     nn.MaxPool2d(kernel_size=2)
        # )

        self.flatten = nn.Flatten()

        self.linear_stack = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout1d(p=0.5),

            nn.LazyLinear(128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout1d(p=0.5),

            # nn.LazyLinear(16),
            # nn.ReLU(),
            # nn.BatchNorm1d(16),
            # nn.Dropout1d(p=0.5),

            # nn.LazyLinear(250),
            # nn.ReLU(),
            # nn.BatchNorm1d(250),
            # nn.Dropout1d(p=0.5),

            # nn.LazyLinear(125),
            # nn.ReLU(),
            # nn.BatchNorm1d(125),
            # nn.Dropout1d(p=0.5),

            # nn.LazyLinear(512),
            # nn.ReLU(),
            # nn.BatchNorm1d(512),
            # nn.Dropout1d(p=0.5),

            # nn.LazyLinear(256),
            # nn.ReLU(),
            # nn.BatchNorm1d(256),

            nn.LazyLinear(2),
        )


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.conv3(out)
        # out = self.conv4(out)
        out = self.flatten(out)
        # print(out.shape)

        out = self.linear_stack(out)
        return out
