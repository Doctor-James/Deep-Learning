# Yolov5环境配置教程

@ powered by Doctor-James

本文章记录本人从零配置yolov5环境，作为一个环境配置教程。

## 1. Anaconda安装

### Python？Anaconda？

装anaconda，就不需要单独装python了

anaconda 是一个python的发行版，包括了python和很多常见的软件库, 和一个包管理器conda，可以通过anaconda在电脑上配置多个python环境，方便不同需求

> 1、anaconda里面集成了很多关于python科学计算的第三方库，主要是安装方便，而python是一个编译器，如果不使用anaconda，那么安装起来会比较痛苦，各个库之间的依赖性就很难连接的很好。
>
> 2、常见的科学计算类的库都包含在里面了，使得安装比常规python安装要容易

安装anaconda建议使用清华镜像站，速度较快,但清华镜像站目前只更新到==v5.3.1==，一些新包用这个版本可能安装不上，所以我们选择先用清华镜像站下载老版本，后续再升级

```SHELL
https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/
```

![屏幕截图](/home/zjl/图片/屏幕截图.png)

下载完成后，运行安装脚本

```shell
bash Anaconda3-5.2.0-Linux-x86_64.sh
```

安装过程中除了问是否安装VSCode选no，其他均选yes或按Enter即可

安装过程中会自动配置环境变量，所以安装完成后只需执行以下命令即可

```shell
source ~/.bashrc
```

最后检验是否安装好

```shell
conda --version
conda list
```

![image-20211003111728832](/home/zjl/.config/Typora/typora-user-images/image-20211003111728832.png)

检测python环境，发现Anaconda已经为我们安装好了

```shel
python
```

![image-20211003111816691](/home/zjl/.config/Typora/typora-user-images/image-20211003111816691.png)

最后进行Anaconda的升级,会自动升级到最新版本

```shell
conda upgrade -n base -c defaults --override-channels conda
```



## 2. 确定需求

首先在github上面git clone Yolov5源码，打开其中==requirements.txt==文件，里面详细写了需要配置的环境版本

```c++
# pip install -r requirements.txt

# Base ----------------------------------------
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python>=4.1.2
Pillow>=7.1.2
PyYAML>=5.3.1
scipy>=1.4.1
torch>=1.7.0
torchvision>=0.8.1
tqdm>=4.41.0

# Logging -------------------------------------
tensorboard>=2.4.1
# wandb

# Plotting ------------------------------------
pandas
seaborn>=0.11.0

# Export --------------------------------------
# coremltools>=4.1  # CoreML export
# onnx>=1.9.0  # ONNX export
# onnx-simplifier>=0.3.6  # ONNX simplifier
# scikit-learn==0.19.2  # CoreML quantization
# tensorflow>=2.4.1  # TFLite export
# tensorflowjs>=3.9.0  # TF.js export

# Extras --------------------------------------
# albumentations>=1.0.3
# Cython  # for pycocotools https://github.com/cocodataset/cocoapi/issues/172
# pycocotools>=2.0  # COCO mAP
# roboflow
thop  # FLOPs computation
```

其中大部分环境都可以用==pip install -r requirements.txt==

其中==torch==和==torchvision==建议自己手动安装（将此文件中==torch==和==torchvision==部分删掉）

为什么我们需要手动安装torch和torchvision呢，是因为实际上这里要求的torch是大于1.7.0，而直接执行这个安装命令会安装为1.7.0，导致程序不能运行，所以需要手动安装更高版本

确定了==torch==版本，还需要确定对应的**显卡驱动**，**CUDA**，**CUDNN**版本，可在PyTorch官网查询对应的版本

```c++
https://pytorch.org/get-started/previous-versions/    //PyTorch官网
```

此处我选择了安装**pytorch v1.8.0**，对应**CUDA 11.1**，**Linux x86_64 显卡驱动版本>=455.23**

## 3. 配置环境

## 3.1 安装显卡驱动

详细教程见此篇知乎

```c++
https://zhuanlan.zhihu.com/p/59618999
```

## 3.2 安装CUDA&cuDNN

* **进入NVIDIA官网下载CUDA**

```c++
https://developer.nvidia.com/cuda-toolkit-archive
```

![2021-10-01 17-36-41 的屏幕截图](/home/zjl/图片/2021-10-01 17-36-41 的屏幕截图.png)

选择合适的版本之后（我这里选择的CUDA 11.1），选择如下图

![2021-10-01 17-36-54 的屏幕截图](/home/zjl/图片/2021-10-01 17-36-54 的屏幕截图.png)

下载完成后运行脚本

```shell
sudo sh cuda_11.1.0_455.23.05_linux.run
```

安装完毕之后，将以下两条加入`.bashrc`文件中

```shell
sudo vim ~/.bashrc

export PATH=/usr/local/cuda-11.1/bin${PATH:+:$PATH}}      #注意，根据自己的版本，修改cuda-11.1...
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} #注意，根据自己的版本，修改cuda-11.1
```

到此为止cuda就安装好了

* **进入NVIDIA官网下载cuDNN（需要注册登陆）**

```c++
https://developer.nvidia.com/rdp/cudnn-archive#a-collapse742-10
```

![2021-10-01 17-35-29 的屏幕截图](/home/zjl/图片/2021-10-01 17-35-29 的屏幕截图.png)

选择适配CUDA版本的cuDNN，我这里选择的是cuDNN v8.0.5

下载下来之后解压

接着复制cuDNN内容到cuda相关文件夹内

```shell
sudo cp cuda/include/cudnn.h    /usr/local/cuda/include      #注意，解压后的文件夹名称为cuda ,将对应文件复制到 /usr/local中的cuda内
sudo cp cuda/lib64/libcudnn*    /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h   /usr/local/cuda/lib64/libcudnn*
```

由于libcudnn*较大，笔者这里遇到了根目录内存不足的问题，遂使用软连接的方法，避免内存重复占用

命令为

```bash
ln  -s  source/*  target/
```

如我把/home/zjl/Downloads/cudnn-11.1-linux-x64-v8.0.5.39/cuda/lib64下所有文件都连接到/usr/local/cuda/lib64/中：

```shell
sudo ln -s /home/zjl/Downloads/cudnn-11.1-linux-x64-v8.0.5.39/cuda/lib64/* /usr/local/cuda/lib64/
```

到此处，CUDA和cuDNN的安装就完成了。

可运行==NVIDIA_CUDA-11.1_Samples==里面的demo检验一下安装

```shell
cd /usr/local/cuda/samples/1_Utilities/deviceQuery #由自己电脑目录决定
make
sudo ./deviceQuery
```

![2021-10-01 22-23-38 的屏幕截图](/home/zjl/图片/2021-10-01 22-23-38 的屏幕截图.png)

出现上述信息，说明cuda配置正确

## 3.3PyTorch环境

首先conda添加清华源，下载速度会比较快

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```

创建一个新的虚拟环境,并命名为==yolov5==（命名随意）

```bash
conda create -n yolov5 python==3.7
source activate yolov5
```

安装PyTorch，torchvision

```bash
conda install pytorch==1.8.0 torchvision==0.9.0
```

最后验证pytorch和torchvision是否安装好

```bash
python
import torch
torch.__version__
import torchvision
torchvision.__version__
```

![image-20211003161156733](/home/zjl/.config/Typora/typora-user-images/image-20211003161156733.png)

## 3.4 安装其他包

在自己下载的yolov5源码目录下（我的是/home/zjl/code/DNN/yolov5）打开终端，注意，此时打开终端系统默认的环境是base环境，base环境是安装anaconda时候conda自动配置的

![image-20211003154317639](/home/zjl/.config/Typora/typora-user-images/image-20211003154317639.png)

而之前我们安装的python，pytorch，torchision都是在自创的虚拟环境**yolov5**中的，所以我们首先需要的是切换到我们自创的虚拟环境中，接着执行安装命令

**注意**：由于我们之前自己手动安装了pytorch和torchvision，所以执行安装命令之前要将==requirements.txt==中的**torch>=1.7.0**，**torchvision>=0.8.1**删除掉

```bash
source activate yolov5
pip install -r requirements.txt
```

到此为之我们所有的环境都已经安装好了

## 4. 运行demo

在yolov5文件夹下执行以下命令测试是否安装完毕

```bash
python detect.py --source data/images/ --weights yolov5s.pt --conf 0.4
```

执行这个命令会自动在官网下载yolov5s.pt文件，如果执行命令下载失败，可以手动到官网下载

```bash
https://github.com/ultralytics/yolov5/releases/tag/v4.0
```

最后如果显示如下，那么恭喜你yolov5安装成功啦！！（也是恭喜我自己。。。）

![image-20211003155201190](/home/zjl/.config/Typora/typora-user-images/image-20211003155201190.png)

demo执行结果如图

![bus](/home/zjl/code/DNN/yolov5/runs/detect/exp4/bus.jpg)

## 5. 结语

众所周知，深度学习配置环境一直是一个很搞心态的事情，但其实只要你心平气和，首先确定自己需要什么，再一步一步的配置，整个配置流程逻辑是很清晰的。

这也是笔者第一次配置这个环境，整个流程用时仅2个多小时。

希望大家以后做事也可以这样首先明确自己的目标，再落到实地，心平气和的一步一步的探索，配置过程中也许会遇到本教程没有出现的问题，希望大家可以静下心来排查问题，这样才可以起到事半功倍的效果。
