import math
import torch.serialization
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
# import datetime
import config

# from utils import utils
from toolbox.arch import Feature_CARB_mutil, Feature_CARB_mutil_small


def normalize(tensorInput):
    tensorRed = (tensorInput[:, 0:1, :, :] - 0.485) / 0.229
    tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
    tensorBlue = (tensorInput[:, 2:3, :, :] - 0.406) / 0.225
    return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)


def denormalize(tensorInput):
    tensorRed = (tensorInput[:, 0:1, :, :] * 0.229) + 0.485
    tensorGreen = (tensorInput[:, 1:2, :, :] * 0.224) + 0.456
    tensorBlue = (tensorInput[:, 2:3, :, :] * 0.225) + 0.406
    return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.feature_small = Feature_CARB_mutil_small(in_ch=9, out_ch=3)
        self.feature_small_ = Feature_CARB_mutil_small(in_ch=9, out_ch=3)

        self.feature = Feature_CARB_mutil(in_ch=6, out_ch=3)

    def forward(self, frames):
        """
        :param frames: [batch_size=1, img_num=3, n_channels=3, h, w]
        :return: img_tensor:
        """
        # print('**', frames.shape)
        for i in range(frames.size(1)):
            frames[:, i, :, :, :] = normalize(frames[:, i, :, :, :])

        x_1 = torch.cat((frames[:, 0, :, :, :], frames[:, 2, :, :, :], frames[:, 4, :, :, :]), dim=1)
        x_0 = torch.cat((frames[:, 1, :, :, :], frames[:, 2, :, :, :], frames[:, 3, :, :, :]), dim=1)
        y_1 = frames[:, 2, :, :, :] - self.feature_small(x_1)
        y_0 = frames[:, 2, :, :, :] - self.feature_small_(x_0)
        # # x = torch.cat((y_1, frames[:, 2, :, :, :], y_0), dim=1)
        x = torch.cat((y_1, y_0), dim=1)


        Img = self.feature(x)

        Img = denormalize(Img)
        # print('Img: ', Img.shape)
        return Img   # precess_time


# https://github.com/ShawnBIT/UNet-family/blob/master/networks/UNet.py
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
