import torch
import torch.nn as nn
from torch.nn import functional as F

"""
Return [bs, 256, 4, 4]
"""


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    
def residual_block(input_channels,output_channels,num_res,down_sample = False):
    block = []
    for i in range(num_res):
        if i == 0 and down_sample:
            block.append(Residual(input_channels,output_channels,use_1x1conv=True,strides=2))
        elif i == 0:
            block.append(Residual(input_channels,output_channels))
        else:
            block.append(Residual(output_channels,output_channels))
    return block

class ResNet_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64),
                       nn.ReLU())
        self.block1 = nn.Sequential(*residual_block(64, 128, 2, True))
        self.block2 = nn.Sequential(*residual_block(128, 256, 2, True))

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)

        return x


if __name__ == "__main__":
    model = ResNet_1()
    print(model)

    # Test with CIFAR-10 size input
    input_tensor = torch.randn(1, 3, 32, 32)
    output = model(input_tensor)
    print(output.shape) 