import os
import datetime
import torch
import matplotlib.pyplot as plt
from Network import Net

from multi_read_data import MemoryFriendlyLoader
import config
from pietorch.pytorch_ssim import SSIM
from tqdm import tqdm

# ------------------------------
plt.switch_backend('agg')

task = config.task
dataset_dir = config.dataset_dir
edited_img_dir = config.edited_img_dir
pathlistfile = config.pathlistfile
visualize_root = config.visualize_root
visualize_pathlist = config.visualize_pathlist
checkpoints_root = config.checkpoints_root
model_besaved_root = config.model_besaved_root
model_best_name = config.model_best_name
model_final_name = config.model_final_name
gpuID = config.gpuID

if task == '':
    raise ValueError('Missing [--task].\nPlease enter the training task.')
elif task not in ['interp', 'denoise', 'denoising', 'sr', 'super-resolution']:
    raise ValueError('Invalid [--task].\nOnly support: [interp, denoise/denoising, sr/super-resolution]')

if dataset_dir == '':
    raise ValueError('Missing [--dataDir].\nPlease provide the directory of the dataset. (Vimeo-90K)')
if task in ['denoise', 'denoising', 'sr', 'super-resolution'] and edited_img_dir == '':
    raise ValueError('Missing [--ex_dataDir]. \
                    \nPlease provide the directory of the edited image dataset \
                    \nif you train on denoising or super resolution task. (Vimeo-90K)')

if pathlistfile == '':
    raise ValueError('Missing [--pathlist].\nPlease provide the pathlist index file.')

if gpuID == None:
    cuda_flag = False
else:
    cuda_flag = True
    torch.cuda.set_device(gpuID)
# --------------------------------------------------------------
LR = config.LR
EPOCH = config.EPOCH
WEIGHT_DECAY = config.WEIGHT_DECAY
BATCH_SIZE = config.BATCH_SIZE
LR_strategy = config.LR_strategy
h = config.h
w = config.w

ssim_weight = config.ssim_weight
l1_loss_weight = config.l1_loss_weight

use_checkpoint = config.use_checkpoint  # 一开始不使用检查点
checkpoint_exited_path = config.checkpoint_exited_path
work_place = config.work_place
model_name = config.model_name
Training_pic_path = config.Training_pic_path
model_information_txt = config.model_information_txt


# --------------------------------------------------------------
Dataset = MemoryFriendlyLoader(origin_img_dir=dataset_dir, edited_img_dir=edited_img_dir, pathlistfile=pathlistfile,
                               task=task)
train_loader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

sample_size = Dataset.count


# --------------------------------------------------------------
# some functions
def show_time(now):
    s = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + '%02d' % now.hour + ':' + '%02d' % now.minute + ':' + '%02d' % now.second
    return s


def delta_time(datetime1, datetime2):
    if datetime1 > datetime2:
        datetime1, datetime2 = datetime2, datetime1
    second = 0
    # second += (datetime2.year - datetime1.year) * 365 * 24 * 3600
    # second += (datetime2.month - datetime1.month) * 30 * 24 * 3600
    second += (datetime2.day - datetime1.day) * 24 * 3600
    second += (datetime2.hour - datetime1.hour) * 3600
    second += (datetime2.minute - datetime1.minute) * 60
    second += (datetime2.second - datetime1.second)
    return second


def save_checkpoint(net, optimizer, epoch, losses, savepath):
    save_json = {
        'cuda_flag': net.cuda_flag,
        'h': net.height,
        'w': net.width,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losses': losses
    }
    torch.save(save_json, savepath)


def load_checkpoint(net, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    net.cuda_flag = checkpoint['cuda_flag']
    net.height = checkpoint['h']
    net.width = checkpoint['w']
    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    losses = checkpoint['losses']

    return net, optimizer, start_epoch, losses


# --------------------------------------------------------------
toflow = Net().cuda()

optimizer = torch.optim.Adam(toflow.parameters(), lr=LR)
schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCH)
ssim_loss = SSIM()
l1_loss = torch.nn.L1Loss()
l2_loss = torch.nn.MSELoss()

# Training
prev_time = datetime.datetime.now()  # current time
print('%s  Start training...' % show_time(prev_time))
plotx = []
ploty = []
start_epoch = 0
check_loss = 10000 

if use_checkpoint:
    print('$_' * 10, 'use checkpoint', '_$' * 10)
    toflow, optimizer, start_epoch, ploty = load_checkpoint(toflow, optimizer, checkpoint_exited_path)
    plotx = list(range(len(ploty)))
    check_loss = min(ploty)

for epoch in range(start_epoch, EPOCH):
    losses_epoch = 0
    print('*' * 20, 'epoch: ', epoch + 1)
    # losses = 0
    count = 0
    step = 0
    for step, (x, y, path_code) in enumerate(tqdm(train_loader)):  # x: b,n,c,h,w=1x9x3xHxW
        ST_LOSS = 0
        LT_LOSS = 0
        VGG_LOSS = 0
        L1_LOSS = 0
        SSIM_LOSS = 0

        losses = 0
        index = 0
        for index in range(2, 3):  # 训练时候中间帧号：1, 2, 3， 4， 5, 6, 7    减少训练帧数, 修改losses ！！！
            tmp = torch.zeros(size=(1, 5, 3, x.size(3), x.size(4)))  # 一次输入连续3帧
            tmp[:, :, :, :, :] = x[:, index - 2:index + 3, :, :, :]
            tmp = tmp.cuda()
            tmp_y = torch.zeros(size=(1, 3, y.size(3), y.size(4)))
            tmp_y[:, :, :, :, ] = y[:, index, :, :, :]
            tmp_y = tmp_y.cuda()

            y_hat = toflow(tmp)

            # print(prediction.shape, tmp_y.shape)
            # loss = l2_loss(prediction, tmp_y)
            l1lossprint = l1_loss(y_hat, tmp_y) * l1_loss_weight
            ssimlossprint = (-ssim_loss(y_hat, tmp_y)) * ssim_weight
            L1_LOSS += l1lossprint
            SSIM_LOSS += ssimlossprint
            # print('l1lossssimloss: ', l1lossprint.item(), ssimlossprint.item(), '\n', '-'*40)

            losses += l1lossprint.item() + ssimlossprint.item()

        ### overall loss
        overall_loss = L1_LOSS + SSIM_LOSS  # + ST_LOSS
        optimizer.zero_grad()
        ### backward loss
        overall_loss.backward()
        ### update parameters
        optimizer.step()
        # print('index: ', index)
        losses /= (index)  # 一个视频中的一帧的平均loss
        # print('losses_pre_%d_video: ' % step, losses)
        losses_epoch += losses  # 所有视频的平均loss的和
    schedular.step()

    print('\n%s  epoch %d: Average_loss=%f\n' % (
    show_time(datetime.datetime.now()), epoch + 1, losses_epoch / (step + 1)))

    # learning rate strategy
    # if epoch in LR_strategy:
    #     optimizer.param_groups[0]['lr'] /= 10

    plotx.append(epoch + 1)
    ploty.append(losses_epoch / (step + 1))
    if epoch // 1 == epoch / 1:
        plt.plot(plotx, ploty)
        plt.savefig(Training_pic_path)

    # checkpoint and then prepare for the next epoch
    if not os.path.exists(checkpoints_root):
        os.mkdir(checkpoints_root)

    # save_checkpoint(toflow, optimizer, epoch + 1, ploty, checkpoints_root + '/checkpoints_%depoch.ckpt' % (epoch + 1))

    if check_loss > losses_epoch / (step + 1):
        print('\n%s Saving the best model temporarily...' % show_time(datetime.datetime.now()))
        if not os.path.exists(os.path.join(work_place, model_besaved_root)):
            os.mkdir(os.path.join(work_place, model_besaved_root))
        torch.save(toflow.state_dict(),
                   os.path.join(work_place, model_besaved_root, model_name + model_best_name), _use_new_zipfile_serialization=False)

        print('Saved.\n')
        check_loss = losses_epoch / (step + 1)
        check_point = losses_epoch / (step + 1)
    else:
        print('\n this epoch is not a better model !')
    if epoch % 10 == 0:
        name = model_name + str(epoch)
        torch.save(toflow.state_dict(), os.path.join(work_place, model_besaved_root, name+config.model_houzhui), _use_new_zipfile_serialization=False)

plt.plot(plotx, ploty)
plt.savefig(Training_pic_path)

cur_time = datetime.datetime.now()
h, remainder = divmod(delta_time(prev_time, cur_time), 3600)
m, s = divmod(remainder, 60)
print('%s  Training costs %02d:%02d:%02d' % (show_time(datetime.datetime.now()), h, m, s))

print('\n%s Saving model...' % show_time(datetime.datetime.now()))
if not os.path.exists(os.path.join(work_place, model_besaved_root)):
    os.mkdir(os.path.join(work_place, model_besaved_root))
# if not os.path.exists(os.path.join(work_place, 'toflow_models_mine')):
#     os.mkdir(os.path.join(work_place, 'toflow_models_mine'))

# save the whole network
# torch.save(toflow, os.path.join(work_place, 'toflow_models', model_name + '.pkl'))

# just save the parameters.
torch.save(toflow.state_dict(), os.path.join(work_place, model_besaved_root, model_name + model_final_name), _use_new_zipfile_serialization=False)
# torch.save(toflow.state_dict(), os.path.join(work_place, 'toflow_models_mine', model_name + '_final_rain25L80_3_params.pkl'))

print('\n%s  Collecting some information...' % show_time(datetime.datetime.now()))
fp = open(os.path.join(work_place, model_besaved_root, model_information_txt), 'w')
fp.write('Model Path:%s\n' % os.path.join(work_place, model_besaved_root, model_name + model_final_name))
# fp = open(os.path.join(work_place, 'toflow_models_mine', model_information_txt), 'w')
# fp.write('Model Path:%s\n' % os.path.join(work_place, 'toflow_models_mine', model_name + '_final_rain25L80_3_params.pkl'))
fp.write('\nModel Structure:\n')
print(toflow, file=fp)
fp.write('\nModel Hyper Parameters:\n')
fp.write('\tEpoch = %d\n' % EPOCH)
fp.write('\tBatch size = %d\n' % BATCH_SIZE)
fp.write('\tLearning rate = %f\n' % LR)
fp.write('\tWeight decay = %f\n' % WEIGHT_DECAY)
print('\tLR strategy = %s' % str(LR_strategy), file=fp)
fp.write('Train on %dK_%s\n' % (int(sample_size / 1000), 'Vimeo'))
print("Training costs %02d:%02d:%02d" % (h, m, s), file=fp)
fp.close()

cur_time = datetime.datetime.now()
h, remainder = divmod(delta_time(prev_time, cur_time), 3600)
m, s = divmod(remainder, 60)
print('%s  Totally costs %02d:%02d:%02d' % (show_time(datetime.datetime.now()), h, m, s))
print('%s  All done.' % show_time(datetime.datetime.now()))
