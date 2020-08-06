# -*- coding: utf-8 -*-
"""
# @file name  : 1_split_dataset.py
# @author     : ae
# @date       : 2019-09-07 10:08:00
# @brief      : 将数据集划分为训练集，验证集，测试集
"""

import os
import random
import shutil


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    random.seed(1)
    # 定义数据集和拆分成的训练机，测试集的路径
    dataset_dir = os.path.join("data", "RMB_data")
    split_dir = os.path.join("data", "RMB_data_split")
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")
    # 80%做训练集，10%做验证集，10%做测试集
    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    for root, dirs, files in os.walk(dataset_dir):  # 遍历数据集路径下的文件，返回的是数据集路径，路径下包含的文件名，文件，都不包含子目录
        for sub_dir in dirs:  # 在两个子文件夹 cat，dog下循环

            imgs = os.listdir(os.path.join(root, sub_dir))  # 以列表形式列出某个子文件夹下的文件名
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            random.shuffle(imgs)  # 打乱顺序
            img_count = len(imgs)  # 记录图像的总数量

            train_point = int(img_count * train_pct)
            valid_point = int(img_count * (train_pct + valid_pct))

            for i in range(img_count):  # 给每张图像确定放的路径位置
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                shutil.copy(src_path, target_path)  # 把原来数据集里的图片拷贝进切分数据集之后的新路径

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point-train_point,
                                                                 img_count-valid_point))
