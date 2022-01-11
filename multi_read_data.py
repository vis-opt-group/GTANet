import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
# from skimage import io
# from skimage import color
# import cv2
import config


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, origin_img_dir, pathlistfile, edited_img_dir='', task=''):
        self.origin_img_dir = origin_img_dir  # 相对应的gtc图片的位置
        self.edited_img_dir = edited_img_dir  # 经过预处理的需要来训练的图片的位置
        self.task = task
        self.pathlist = self.loadpath(pathlistfile)
        self.count = len(self.pathlist)

    def loadpath(self, pathlistfile):
        fp = open(pathlistfile)
        pathlist = fp.read().splitlines()
        fp.close()
        return pathlist  # 返回 pathlistfile 文件如 train_mine.txt 中的字符串路径构成的列表list

    def __getitem__(self, index):
        frames = []
        frames_gtc = []
        path_code = self.pathlist[index]  # index指示的文件夹 如 00006/10
        N = config.N  # 7张雨图，以及rfc-4
        for i in range(1, 6):  # edited_img_dir/path_code 某个雨图文件夹
            frames.append((plt.imread(os.path.join(self.edited_img_dir, path_code, 'rfc-%d.jpg' % (i)))[0:config.h,
                           0:config.w] / 255.0))  # [0:config.h, 0:config.w]
            frames_gtc.append((plt.imread(os.path.join(self.origin_img_dir, path_code, 'gtc-%d.jpg' % (i)))[0:config.h,
                               0:config.w] / 255.0))

        frames = np.asarray(frames, dtype=np.float32)
        frames_gtc = np.asarray(frames_gtc, dtype=np.float32)
        framex = np.transpose(frames[:, :, :, :], (0, 3, 1, 2))
        framey = np.transpose(frames_gtc[:, :, :, :], (0, 3, 1, 2))

        return torch.from_numpy(framex), torch.from_numpy(framey), path_code

    def __len__(self):
        return self.count
