import torch
from torch import nn
from torch import autograd
from torch.nn import functional as F
from torch.autograd import Variable


class Feature(nn.Module):
    def __init__(self, in_channel):
        super(Feature, self).__init__()

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=4, dilation=4)
        self.fusion = nn.Sequential(nn.Conv2d(in_channels=64 * 3, out_channels=64, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        feature_1 = self.conv1(x)
        feature_2 = self.conv2(x)
        feature_3 = self.conv3(x)

        inputs = torch.cat((feature_1, feature_2, feature_3), 1)
        fusion_outputs = self.fusion(inputs)

        return fusion_outputs
