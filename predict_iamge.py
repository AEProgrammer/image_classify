import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import random
from PIL import Image
import torchvision
import torchvision.transforms as transforms


def load_pre():
    model = torchvision.models.resnet34(pretrained=False)
    fc_inchannel = model.fc.in_features
    model.fc = nn.Linear(fc_inchannel, 2)

    model_weight_path = "save_model/resnet_epoch1.pth"
    model.load_state_dict(torch.load(model_weight_path))
    # print(model.fc.in_features)
    return model

net = load_pre()

# 预测图片的路径 并给图片一些转换操作，让他成为224*224的图像并转化为tensor再给数值进行标准化处理
img = Image.open("data/predict_image/test1.jpeg").convert('RGB')
plt.imshow(img)
plt.show()

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# 如果使用的网络是lenet下面的resize要(32,32), 如果使用的网络是mynet下面resize为(224,224)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

img = test_transform(img)
img = img.unsqueeze(0)  # 模型接收的是batch_size*channels*h*w的输入，给img增加一个batch_size为1的维度
print(img.shape)

# 开始预测分类
net.eval()
with torch.no_grad():
    outputs = net(img)
    print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    if predicted.numpy().item() == 1:  # 如果加载的是模型参数把这里数字设置为0，如果加载的是模型数字设置为1
        print("its a cat")
    else:
        print("its a dog")





