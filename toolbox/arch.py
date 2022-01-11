# import math
import torch
import torch.serialization
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import numpy as np
from lib.se_nets import SELayer


class ResBlock(torch.nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, frames):
        """
        Args:
            frames: 1x64xHxW

        Returns: 1x64xHxW

        """
        res = self.conv1(frames)
        res = torch.nn.functional.relu(res)
        res = self.conv2(res)
        return frames + res


class Feature(torch.nn.Module):

    def __init__(self):
        super(Feature, self).__init__()
        self.preconv = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.resblock_1 = ResBlock()
        self.resblock_2 = ResBlock()
        self.resblock_3 = ResBlock()
        self.resblock_4 = ResBlock()
        self.resblock_5 = ResBlock()
        self.conv1x1 = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

    def forward(self, frame):
        """
        Args:
            frame: 1x3xHxW

        Returns: 1x3xHxW

        """
        x = self.preconv(frame)
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        x = self.resblock_4(x)
        x = self.resblock_5(x)
        x = self.conv1x1(x)
        return frame - x


class CARB(torch.nn.Module):

    def __init__(self, ch=128):
        super(CARB, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1)
        self.se_attention = SELayer(channel=ch)
        self.conv_256_128_1x1 = torch.nn.Conv2d(in_channels=ch+ch, out_channels=ch, kernel_size=1)

    def forward(self, feature):
        """
        Args:
            feature: 1x128xHxW

        Returns: 1x128xHxW

        """
        x = self.conv1(feature)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x_se = self.se_attention(x)
        f = torch.cat([x, x_se], dim=1)
        f = self.conv_256_128_1x1(f)
        return feature + f


class Feature_CARB(torch.nn.Module):
    def __init__(self):
        super(Feature_CARB, self).__init__()
        self.preconv = torch.nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.resblock_1 = CARB()
        self.resblock_2 = CARB()
        self.resblock_3 = CARB()
        self.resblock_4 = CARB()
        self.resblock_5 = CARB()
        self.postconv = torch.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.ps = torch.nn.PixelShuffle(upscale_factor=2)
        self.conv_1x1 = torch.nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1)

    def forward(self, frame):
        """
        Args:
            frame: 1x3xHxW

        Returns: 1x3xHxW

        """
        x = self.preconv(frame)
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        x = self.resblock_4(x)
        x = self.resblock_5(x)
        x = self.postconv(x)
        x = self.ps(x)
        x = self.conv_1x1(x)
        return frame - x


class Feature_CARB_mutil(torch.nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(Feature_CARB_mutil, self).__init__()
        self.preconv = torch.nn.Conv2d(in_channels=in_ch, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.preconv = torch.nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.resblock_1 = CARB()
        self.resblock_2 = CARB()
        self.resblock_3 = CARB()
        self.resblock_4 = CARB()
        self.resblock_5 = CARB()
        # self.postconv = torch.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.ps = torch.nn.PixelShuffle(upscale_factor=2)
        # self.conv_3x3 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        # self.conv_1x1 = torch.nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=1)
        self.conv_3x3 = torch.nn.Conv2d(in_channels=128, out_channels=out_ch, kernel_size=3, padding=1)

    def forward(self, frame):
        """
        Args:
            frame: 1xcxHxW

        Returns: 1x3xHxW

        """
        # mid = frame[:, 3:6, :, :]
        x = self.preconv(frame)
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        x = self.resblock_4(x)
        x = self.resblock_5(x)
        # x = self.postconv(x)
        # x = self.ps(x)
        x = self.conv_3x3(x)
        # x = self.conv_1x1(x)
        # return frame - x
        # return mid - x
        return x


class Feature_CARB_mutil_small(torch.nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(Feature_CARB_mutil_small, self).__init__()
        self.preconv = torch.nn.Conv2d(in_channels=in_ch, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.preconv = torch.nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.resblock_1 = CARB()
        self.resblock_2 = CARB()
        # self.resblock_3 = CARB()
        # self.resblock_4 = CARB()
        # self.resblock_5 = CARB()
        # self.postconv = torch.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.ps = torch.nn.PixelShuffle(upscale_factor=2)
        self.conv_3x3 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv_1x1 = torch.nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=1)

    def forward(self, frame):
        """
        Args:
            frame: 1xcxHxW

        Returns: 1x3xHxW

        """
        # mid = frame[:, 3:6, :, :]
        x = self.preconv(frame)
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        # x = self.resblock_3(x)
        # x = self.resblock_4(x)
        # x = self.resblock_5(x)
        # x = self.postconv(x)
        # x = self.ps(x)
        x = self.conv_3x3(x)
        x = self.conv_1x1(x)
        # return frame - x
        return x

# https://github.com/ShawnBIT/UNet-family/blob/master/networks/UNet.py
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
