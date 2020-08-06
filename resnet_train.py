import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from my_dataset import CatDogDataset
import torchvision


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(1)

# 参数设置
MAX_EPOCH = 1
BATCH_SIZE = 128
LR = 0.0001
log_interval = 5
val_interval = 1

# ======================= step 1/5 数据 ===================

# 设置数据集路径
split_dir = os.path.join("data", "cat_dog_split")
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")
test_dir = os.path.join(split_dir, "test")

# 设置数据预处理和数据增强方法
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([transforms.RandomCrop(200, padding=24, padding_mode='reflect')], p=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# 构建dataset
train_data = CatDogDataset(data_dir=train_dir, transform=train_transform)
valid_data = CatDogDataset(data_dir=valid_dir, transform=test_transform)
test_data = CatDogDataset(data_dir=test_dir, transform=test_transform)

# 构建dataloader
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  # shuffle代表乱序
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

# ==================step 2/5 模型================
def load_pre():
    model = torchvision.models.resnet34(pretrained=False)
    model_weight_path = "save_model/resnet34_pre.pth"

    model.load_state_dict(torch.load(model_weight_path))
    # print(model.fc.in_features)
    fc_inchannel = model.fc.in_features
    model.fc = nn.Linear(fc_inchannel, 2)
    return model
net = load_pre()

# =================step 3/5 损失函数=============
criterion = nn.CrossEntropyLoss()

# =================step 4/5 优化器==============
optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ==================step 5/5 训练===============
total_step = len(train_loader)

for epoch in range(MAX_EPOCH):
    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train()
    for i, data in enumerate(train_loader):

        # forward
        inputs, labels = data
        # print(inputs.shape)
        outputs = net(inputs)

        # backward
        optimizer.zero_grad()  # 先把梯度置0
        loss = criterion(outputs, labels)  # 用交叉熵损失函数计算损失
        loss.backward()  # 反向传播计算梯度

        # updata weights
        optimizer.step()  # 用定义好的优化器更新参数

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i + 1, total_step, loss_mean, correct / total))
            loss_mean = 0.

    scheduler.step()

    if (epoch+1) % val_interval == 0:
        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):

                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # 统计分类情况
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().numpy()
                loss_val += loss.item()

            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j + 1, len(valid_loader), loss_val / len(valid_loader), correct_val / total_val))

    torch.save(net.state_dict(), "save_model/"+"mynet_epoch"+str(epoch)+'.pth')
