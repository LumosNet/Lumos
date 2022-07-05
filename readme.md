<div align="center">
  <img src="https://github.com/LumosNet/Lumos/blob/master/img/l3.png">
</div>

# Lumos

### 简介

[Lumos](https://gitee.com/lumos-net/lumos)是一个以个人学习为目的而开发的深度学习框架，以darknet框架作为主要参考。Lumos框架以C语言为主要开发语言，gcc作为主要编译器，当然该框架也可以轻松的在Windows上进行开发。

Lumos一词来源《哈利波特》中的咒语荧光闪烁，该框架旨在提供一个简洁的轻量级深度学习框架，帮助学习、研究深度学习算法和底层实现的学者和爱好者更容易学习相关内容，希望该框架是你学习过程中指引你的荧光。

Lumos的目标从来不是为了比肩TensorFlow或者Pytorch这样的顶级开源框架，而是希望更好的展现底层算法实现，提供给使用者更灵活的使用体验。同时希望有更多的人不止能在上层框架构建成熟应用，更能对底层算法原理和实现技巧有更多的关注。



### 开发状况

当前Lumos提供基础的全连接神经网络和卷积神经网络基本组件，使用配置文件的方式提供快速构建神经网络的方法。如果你有Darknet框架的使用经验，那么你可以快速上手Lumos。

Lumos框架现在只提供了最基本的网络组件，对于一些特定算法的处理方法会在后续开发继续完善。

当前该框架还未能提供GPU加速，你所构建的所有内容都将在CPU上进行计算。



### 安装

Lumos在[Linux](https://www.linux.org/)上开发，我们推荐您在Linux上使用该框架，当然我们也会示范如何在Windows上使用

为了您更方便的获取我们的代码，推荐您安装[Git](https://git-scm.com/)版本控制工具

**Ubuntu：**

首先您需要相关的编译工具，gcc编译器和make工具是您必须安装的工具

```shell
$ sudo apt update
$ sudo apt install build-essential
```

拉取我们的代码，并进入项目目录

```shell
$ git clone git@gitee.com:lumos-net/lumos.git
$ cd lumos
```

现在我们来验证您是否可以正常使用，我们为您准备了一个简单的demo

首先您需要修改根目录下makefile第36行，改为如下

```makefile
EXECOBJA=xor.o
```

其次您需要修改cfg/xor.cfg文件第2行，改为如下

```makefile
batch=1
```

然后，请回到lumos根目录下，命令行输入

```shell
make
```

开始编译lumos

当您编译完成后，命令行输入

```shell
./main.exe ./demo/weights.w
```

如果有如下输出，那么您可以正常使用lumos

```shell
XOR
index  type   filters   ksize        input                  output
  1    im2col                         2 x   1 x   1   ->     1 x 2
  2    connect              4         1 x   2         ->     1 x   4
  3    connect              1         1 x   4         ->     1 x   1
  4    mse                  1         1 x   1         ->     1 x   1
4
xor: [0, 0], test: 0.049111
xor: [1, 1], test: 0.057524
xor: [1, 0], test: 0.960859
xor: [0, 1], test: 0.983461
```



**Windows：**

由于lumos支持CUDA加速，所以我们无法在windows上直接编译，我们推荐您使用wsl2，这是windows推出的Linux Subsystem，并且wsl2支持GPU，相关安装方法请参考[wsl文档](https://docs.microsoft.com/zh-cn/windows/wsl/)

如果您成功安装wsl2，那么请参考上文Linux下Lumos安装方法



**CUDA：**

Lumos支持cuda加速（虽然现在还没有），如果您拥有支持cuda的GPU，并且希望使用GPU加速您的算法，那么请提前安装cuda，相关安装方法请参考[cuda文档](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)



### 发行版

**[v0.1-a](https://gitee.com/lumos-net/lumos/tree/v0.1-a/)**：Lumos第一个测试版本，主要对基本网络框架进行测试，若无必要，请不要使用该版本。当前我们推荐			   您直接使用master分支的最新提交版本

