import torch
from torch import nn
from torch import autograd
from torch.nn import functional as F
from torch.autograd import Variable


class Feature(nn.Module):
    def __init__(self, in_channel):
        super(Feature, self).__init__()

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=4, dilation=4)
        self.fusion1 = nn.Sequential(nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0))
        self.fusion2 = nn.Sequential(nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        feature_1 = self.conv1(x)
        feature_2 = self.conv2(x)
        feature_3 = self.conv3(x)
        feature_4 = self.conv4(x)

        input1 = torch.cat((feature_1, feature_2), 1)
        output1 = self.fusion1(input1)
        input2 = torch.cat((feature_3, feature_4), 1)
        output2 = self.fusion1(input2)
        output = torch.cat((output1, output2), 1)
        fusion_outputs = self.fusion2(output)

        return fusion_outputs
