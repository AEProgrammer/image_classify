# image_classify
## 数据
先在目录下新建data文件夹，将自己的数据集放进data文件夹里，自己的数据集遵循这样一个目录结构<br>
---dataset_name<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---data_class1<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--xxx.jpg/png<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--xxx.jpg/png<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---data_class2<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---data_class3..<br>
在数据集文件夹下以类别名为子文件夹的名字，在每个子文件夹下包含该类别的所有图片<br>
得到数据集后，运行spilt_data.py,注意把数据集路径修改为自己数据集的路径，这样会在原数据集文件的同级目录生成train和val文件夹，里面同样以类别名做为子文件夹，我们以90%做为训练集。<br>
## 训练
运行resnet_train.py进行训练，注意把train_dir和valid_dir路径改成自己的，脚本使用resnet34网络，把最后一层全连接层替换成了两个类别的输出
##预测
运行predict_image.py进行预测，脚本使用resnet34网络，把最后一层全连接层替换成了两个类别的输出，加载自己预训练的模型参数在save_model里，注意这里预测的是猫狗二分类任务。

**如果使用自定义的数据集进行训练以及预测要相应修改resnet_train.py把resnet网络的最后一层换成自己需要的类别个数的输出，resnet默认输出是1000类别，我这里因为是猫狗分类所以修改成了两个类别的输出<br>
同样在predict时也要修改相应的模型，并加载自己预训练的模型参数，然后使用其他数据集要在my_dataset，predict中修改类别与数字的对应关系字典例如{"猫":1,"狗":0}**
