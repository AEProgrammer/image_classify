# -*- coding: utf-8 -*-
import os
from shutil import copy
import random

# 定义一个创建文件夹函数
def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

# 定义数据集路径
file = 'data/cat_dog_data'

# 根据子文件夹名字得到动物类别名的列表 [cat, dog]
animal_classes = [cla for cla in os.listdir(file) if "." not in cla]

# 为每个类别建立训练集文件夹
mkfile('data/cat_dog_split/train')
for cla in animal_classes:
    mkfile('data/cat_dog_split/train/'+cla)

# 为每个类别建立验证集文件夹
mkfile('data/cat_dog_split/valid')
for cla in animal_classes:
    mkfile('data/cat_dog_split/valid/'+cla)

# 为每个类别建立测试集文件夹
mkfile('data/cat_dog_split/test')
for cla in animal_classes:
    mkfile('data/cat_dog_split/test/'+cla)

# 数据集划分比例为10%做验证集，10%做测试集，80%做训练集
split_rate = [0.1, 0.1]
for cla in animal_classes:
    cla_path = file + '/' + cla + '/'
    # images是包含了数据集中某一类别的所有数据名列表
    images = os.listdir(cla_path)
    random.shuffle(images)
    num_images = len(images)
    num_valid = int(num_images * split_rate[0])
    num_test = int(num_images * (split_rate[0] + split_rate[1]))

    for i in range(num_images):
        if i < num_valid:
            image_path = cla_path + images[i]
            new_path = "data/cat_dog_split/valid/" + cla
            copy(image_path, new_path)

        elif i < num_test:
            image_path = cla_path + images[i]
            new_path = "data/cat_dog_split/test/" + cla
            copy(image_path, new_path)

        else:
            image_path = cla_path + images[i]
            new_path = "data/cat_dog_split/train/" + cla
            copy(image_path, new_path)

print('Class:{}, train:{}, valid:{}, test:{}'.format(len(animal_classes), num_images-num_test, num_valid,
                                                                 num_test-num_valid))
