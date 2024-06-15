<div align="center">
  <img src="https://github.com/LumosNet/Lumos/blob/master/img/Lumos.png">
</div>

# Lumos

## 简介

[Lumos](https://github.com/LumosNet/Lumos)是一个简洁的轻量级深度学习框架，帮助学习、研究深度学习算法和底层实现的学者和爱好者更容易学习相关内容，希望该框架是你学习过程中指引你的荧光。

Lumos的目标从来不是为了比肩TensorFlow或者Pytorch这样的顶级开源框架，而是希望更好的展现底层算法实现，提供给使用者更灵活的使用体验。同时希望有更多的人不止能在上层框架构建成熟应用，更能对底层算法原理和实现技巧有更多的关注。

| 说明     | 链接                                                         |
| -------- | ------------------------------------------------------------ |
| 使用手册 | [<img src="https://img.shields.io/badge/Lumos-Docs-brightgreen" />](https://lumos-docs.readthedocs.io/en/latest/) |

## 依赖

Lumos使用C语言开发，并且依赖于CUDA框架实现GPU加速，所以您需要独立安装编译环境和CUDA

您需要安装gcc/g++编译器以及make工具，您可以参考其官方文档

我们还依赖于CUDA Toolkit进行GPU加速，您需要提前安装CUDA，详细安装方法请参考[NVIDIA CUDA官方文档](*https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html*)

## 安装

您需要下载我们提供的安装包，请您下载您所需要的版本，最新版本安装包为

| 说明 | 链接                                                         |
| ---- | ------------------------------------------------------------ |
| v1.0 | [<img src="https://img.shields.io/badge/Lumos-Install-brightgreen" />](https://github.com/LumosNet/Lumos-Build/archive/refs/tags/v1.0.zip) |

下载完成后使用如下命令进行安装

```shell
bash lumos-v1.0.run
```



## 环境设置

我们需要为Lumos进行环境配置，Lumos默认安装于用户目录下，请在~/.bashrc文件中添加如下内容

```shell
export PATH=/home/用户名/lumos/include/:$PATH
export PATH=/home/用户名/lumos/bin:$PATH
export LD_LIBRARY_PATH=/home/用户名/lumos/lib:$LD_LIBRARY_PATH
```

添加完成后使用如下命令激活

```shell
source ~/.bashrc
```

并使用

```shell
lumos --version
```




## 快速入门

Lumos允许您快速实现深度学习模型，我们提供了简洁的接口，您可以在lumos/include目录下的lumos.h文件中查看我们提供的接口

下面我们将用Lenet5模型实现MNIST手写数字识别，让您快速了解Lumos框架的使用

完整的框架教程，请您参考[Lumos教程](https://lumos-docs.readthedocs.io/en/latest/docs/教程/index.html)



### 模型构建

我们通常将一个深度学习模型视为一个计算图，所以在Lumos中一个深度学习模型就是一个计算图，我们需要首先创建一个计算图类的实例

```c
Graph *g = create_graph()
```

在此之后您需要创建不同的计算层，并确定它们的链接方式

```c
Layer *l1 = make_convolutional_layer(6, 5, 1, 0, 1, "relu");
Layer *l2 = make_avgpool_layer(2, 2, 0);
Layer *l3 = make_convolutional_layer(16, 5, 1, 0, 1, "relu");
Layer *l4 = make_avgpool_layer(2, 2, 0);
Layer *l5 = make_convolutional_layer(120, 5, 1, 0, 1, "relu");
Layer *l6 = make_im2col_layer();
Layer *l7 = make_connect_layer(84, 1, "relu");
Layer *l8 = make_connect_layer(10, 1, "relu");
Layer *l9 = make_softmax_layer(10);
Layer *l10 = make_mse_layer(10);
```

```c
append_layer2grpah(g, l1);
append_layer2grpah(g, l2);
append_layer2grpah(g, l3);
append_layer2grpah(g, l4);
append_layer2grpah(g, l5);
append_layer2grpah(g, l6);
append_layer2grpah(g, l7);
append_layer2grpah(g, l8);
append_layer2grpah(g, l9);
append_layer2grpah(g, l10);
```

append_layer2grpah将您创建的计算层按顺序添加到计算图中，此时我们创建的计算图g，就是一个完整的静态深度学习模型

完成模型创建后，我们需要调度模型进行计算，Lumos提供Session会话类来完成全部的计算调度，首先我们需要实例化一个会话

```c
Session *sess = create_session(g, 32, 32, 1, 10, type, path);
```

并设置训练超参数

```c
set_train_params(sess, 15, 16, 16, 0.1);
```

在训练开始前，Lumos需要完成内存等训练环境初始化

```c
init_session(sess, "./data/mnist/train.txt", "./data/mnist/train_label.txt");
```

现在一切准备就绪，可以开始训练了

```c
train(sess);
```



### 编译

上述过程我们完成了模型代码，现在我们需要编译我们的代码，我们推荐您使用gcc编译器，因为Lumos框架在Linux下开发并发布，全面依赖gcc编译器特性

如下我们提供了一个编译命令的参考实例，如果您对make工具和gcc较为熟悉，请您自行编写编译脚本

```shell
gcc -fopenmp lumos.c -I/home/用户名/lumos/include/ -o lumos -L/home/用户名/lumos/lib -llumos
```




## 联系我们：

####     邮箱：yuzhuoandlb@foxmail.com

​    