# -*- coding: utf-8 -*-
"""
# @file name  : dataset.py
# @author     : ae
# @date       : 2019-08-21 10:08:00
# @brief      : 各数据集的Dataset定义
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
animal_label = {"cat": 0, "dog": 1}


class CatDogDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        猫狗分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"cat": 0, "dog": 1}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]  # 从data_info列表中取出index下标对应的路径和标签元组
        img = Image.open(path_img).convert('RGB')     # 0~255， 由路径打开这个路径下的图片并转为rgb

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label  # 返回index索引的图片和标签

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别 子文件夹名字就是类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))  # 得到照片的文件名列表
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):  # 得到文件名列表中每个文件名对应的具体文件路径，再通过子文件夹名字得到label
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = animal_label[sub_dir]
                    data_info.append((path_img, int(label)))  # 把一个图片的路径和标签做为一个元组增添到data_info列表

        return data_info  # 循环结束之后得到data_dir下每张照片的路径和标签组成的列表
