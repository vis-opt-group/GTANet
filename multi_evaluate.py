import torch
import torch.serialization
# from torchvision import transforms
import numpy as np
# import sys
# import getopt
import os
# import shutil 
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg  # 读取图片
# from skimage import io
# from skimage import color

# import datetime
# from PIL import Image
from Network import Net
import warnings
import config
from multi_read_data_eval import MemoryFriendlyLoader

warnings.filterwarnings("ignore", module="matplotlib.pyplot")
# ------------------------------
# I don't know whether you have a GPU.
plt.switch_backend('agg')

task = config.task  # 测试任务的类别
dataset_dir = config.dataset_dir  # 测试数据集
dataset_gtc_dir = config.dataset_gtc_dir  # 测试用到的gtc文件的路径
out_img_dir = config.out_img_dir  # 测试结果存放位置
pathlistfile = config.pathlistfile  # 具体测试用例名称表
model_path = config.model_path  # 要测试的模型位置
gpuID = config.gpuID
BATCHSIZE = config.BATCH_SIZE
h = config.h
w = config.w
N = config.N
map_location = config.map_location

if task == '':
    raise ValueError('Missing [--task].\nPlease enter the training task.')
elif task not in ['interp', 'denoise', 'denoising', 'sr', 'super-resolution']:
    raise ValueError('Invalid [--task].\nOnly support: [interp, denoise/denoising, sr/super-resolution]')

if dataset_dir == '':
    raise ValueError('Missing [--dataDir].\nPlease provide the directory of the dataset. (Vimeo-90K)')

if pathlistfile == '':
    raise ValueError('Missing [--pathlist].\nPlease provide the pathlist index file for test.')

if model_path == '':
    raise ValueError('Missing [--model model_path].\nPlease provide the path of the toflow model.')

if gpuID is None:
    cuda_flag = False
else:
    cuda_flag = True
    torch.cuda.set_device(gpuID)


# --------------------------------------------------------------

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def vimeo_evaluate(input_dir, dataset_gtc_dir, out_img_dir, test_codelistfile, task='', cuda_flag=True):
    mkdir_if_not_exist(out_img_dir)
    # prepare DataLoader
    Dataset = MemoryFriendlyLoader(origin_img_dir=dataset_gtc_dir, edited_img_dir=input_dir, pathlistfile=pathlistfile,
                                   task=task)
    train_loader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=0)
    sample_size = Dataset.count

    net = Net()
    net.load_state_dict(torch.load(model_path, map_location=map_location))

    if cuda_flag:
        net.cuda().eval()
    else:
        net.eval()

    with torch.no_grad():
        for step, (x, path_code) in enumerate(train_loader):
            # x = x.cuda()    # X: 1x31xCxHxW [0:30]]共31帧
            print('x.shape: ', x.shape, x.size(1))
            processing_time_for_onevideo = 0
            for center in range(2, 29):  # 使用连续3帧生成中间帧去雨的结果 [1, 2, 3, 4, 5, ..., 27, 28, 29]
                tmp = torch.zeros(size=(1, 5, 3, x.size(3), x.size(4)))
                tmp[:, :, :, :, :] = x[:, center - 2:center + 3, :, :, :]
                tmp = tmp.cuda()    # 直接将x送进网络, 来进行迭代是有问题的

                predicted_img = net(tmp)  # 现在是 1x3xHxW

                img_ndarray = predicted_img.cpu().detach().numpy()
                img_ndarray = np.transpose(img_ndarray, (0, 2, 3, 1))
                img_ndarray = img_ndarray[0]

                img_tobesaved = np.asarray(img_ndarray)
                mkdir_if_not_exist(os.path.join(out_img_dir, path_code[0]))
                # video = path_code[0].split('/')[0]  # print(path_code)    # ('00001/6',)
                # sep = path_code[0].split('/')[1]
                # plt.imsave(os.path.join(out_img_dir, path_code[0], '%d.jpg' % (center+1)), np.clip(img_ndarray, 0.0, 1.0))
                plt.imsave(os.path.join(out_img_dir, path_code[0], 'result024-%d.jpg' % (center+1)),
                           np.clip(img_tobesaved, 0.0, 1.0))  # 路径有问题
            processing_time_for_onevideo = processing_time_for_onevideo / 29
            print('processing_time_mean for %s-th video :' % path_code, processing_time_for_onevideo, 'us')

        print('*' * 40)
        print('END')


vimeo_evaluate(dataset_dir, dataset_gtc_dir, out_img_dir, pathlistfile, task=task, cuda_flag=cuda_flag)