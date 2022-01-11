import torch
import os
from torch.utils import data
from PIL import Image
from torchvision import transforms


class TrainDataSetDefine(data.Dataset):
    def __init__(self, img_list, label_list, attention_list, config):
        self.img_list = img_list
        self.label_list = label_list
        self.attention_list = attention_list
        self.cfg = config

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.cfg.train_dir, self.img_list[index])
        label_path = os.path.join(self.cfg.train_dir, self.label_list[index])
        attention_path = os.path.join(self.cfg.train_dir, self.attention_list[index])

        with Image.open(img_path) as img:
            img = img.convert('RGB')
        img = self.transform(img)

        with Image.open(label_path) as label:
            label = label.convert('RGB')
        label = self.transform(label)

        with Image.open(attention_path) as attention:
            attention = attention.convert('RGB')
        attention = self.transform(attention)[0:1, :, :]
        print(attention)
        assert img.size() == label.size()
        return img, label, attention, img_name

    def __len__(self):
        len_0 = len(self.img_list)
        len_1 = len(self.label_list)
        len_2 = len(self.attention_list)

        assert len_0 == len_1
        return len_0


class DataSet(object):
    def __init__(self, config):
        self.cfg = config

        self.train_img_list, self.train_label_list, self.attention_list = self.get_list(self.cfg.train_dir)

        self.train_dataset = TrainDataSetDefine(self.train_img_list, self.train_label_list, self.attention_list,
                                                self.cfg)
        self.train_loader = data.DataLoader(dataset=self.train_dataset,
                                            batch_size=self.cfg.train_batch_size,
                                            shuffle=False, )

    @staticmethod
    def get_list(dir):
        img_list = []
        label_list = []
        attention_list = []

        name_list = os.listdir(dir)
        for name in name_list:
            if 'no' not in name and 'at' not in name:
                img_list.append(name)
                label_list.append('no' + name)
                attention_list.append('at' + name)
        return img_list, label_list, attention_list
