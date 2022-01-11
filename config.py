# if train_or_eval = True then 训练 else 测试
train_or_eval = False
# train_or_eval = True

if train_or_eval is not True:
    # 测试的配置
    task = 'denoising'
    dataset_dir = r'/user51/mxy/data/J4R/frames_heavy_test_JPEG'  # 测试图片包括边缘图的路径
    dataset_gtc_dir = r'/user51/mxy/data/J4R/frames_heavy_test_JPEG'
    # 相对应的gtc路径(使用了训练集中的gtc)，所有设置的路径都是00006/1这种文件夹的父文件夹才可以
    out_img_dir = 'evaluate'  # 实验结果存放位置
    pathlistfile = r'/user51/mxy/data/J4R/test_heavy.txt'  # 测试的图片的具体路径
    # pathlistfile = './train/test_heavy.txt'
    model_path = 'toflow_models_mine/denoising_best.pkl'  
    gpuID = 0  # map_location='cuda:1' 
    map_location = 'cuda:0'
    BATCH_SIZE = 1
    h = 888
    w = 888
    N = 3  # 5张图片

else:
    # 训练的配置
    task = 'denoising'
    edited_img_dir = r'/user51/mxy/data/J4R/heavy_train'  # 训练输入的图片的文件夹
    dataset_dir = r'/user51/mxy/data/J4R/heavy_train'
    pathlistfile = r'/user51/mxy/data/J4R/train_heavy.txt'  # 训练的图片的具体路径
    visualize_root = './visualization_mine/'  # 存放展示结果的文件夹
    visualize_pathlist = ['00001/4']  # 需要展示训练结果的训练图片所在的小文件夹
    checkpoints_root = './checkpoints_mine'  # 训练过程中产生的检查点的存放位置
    model_besaved_root = './toflow_models_mine'  # best_model 和 final_model 的参数的保存位置
    model_best_name = '_best.pkl'
    model_final_name = '_final.pkl'
    gpuID = 0

    # Hyper Parameters
    if task == 'interp':
        LR = 3 * 1e-5
    elif task in ['denoise', 'denoising', 'sr', 'super-resolution']:
        # LR = 1 * 1e-5
        LR = 0.0001
    EPOCH = 141
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 1
    LR_strategy = []
    # h = 360
    # w = 480
    h = 256
    w = 256
    # h = 120
    # w = 120
    N = 3  # 输入7张图片

    ssim_weight = 1.1
    l1_loss_weight = 0.75
    w_VGG = 0
    w_ST = 1
    w_LT = 0
    alpha = 50

    use_checkpoint = False  # 一开始不使用已有的检查点
    checkpoint_exited_path = './checkpoints_mine/checkpoints_40epoch.ckpt'  # 已有的检查点
    work_place = '.'
    model_name = task
    model_houzhui = '.pkl'
    Training_pic_path = 'toflow_models_mine/Training_result_mine_maxoper.jpg'
    model_information_txt = model_name + '_information.txt'
