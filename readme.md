<div align="center">
  <img src="https://github.com/LumosNet/Lumos/blob/master/img/Lumos.png">
</div>

# Lumos

## 简介

[Lumos](https://gitee.com/lumos-net/lumos)是一个简洁的轻量级深度学习框架，帮助学习、研究深度学习算法和底层实现的学者和爱好者更容易学习相关内容，希望该框架是你学习过程中指引你的荧光。

Lumos的目标从来不是为了比肩TensorFlow或者Pytorch这样的顶级开源框架，而是希望更好的展现底层算法实现，提供给使用者更灵活的使用体验。同时希望有更多的人不止能在上层框架构建成熟应用，更能对底层算法原理和实现技巧有更多的关注。

| 说明     | 链接                                                         |
| -------- | ------------------------------------------------------------ |
| 使用手册 | [<img src="https://img.shields.io/badge/Lumos-U-brightgreen" />] |



## 安装

Lumos不提供任何安装包，您需要直接编译源代码来使用
```shell
$ git clone https://github.com/LumosNet/Lumos.git
```
我们推荐您使用最新版本代码，或main分支代码

编译Lumos需要使用C/C++编译器，我们推荐您在Linux系统中使用gcc/g++编译器进行编译
您需要提前安装CUDA，详细安装方法请参考[NVIDIA CUDA官方文档](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)



## 编译

首先您需要修改我们为您提供的编译脚本makefile，在编译脚本59行
```shell
LDFLAGS+= -L -lcudart -lcublas -lcurand
```
您需要在-L后添加您cuda的静态链接库文件目录，如下
```shell
LDFLAGS+= -L/usr/local/cuda/lib -lcudart -lcublas -lcurand
```
完成修改后，在命令行使用编译命令make，进行编译




## 测试

编译完成后，您可以通过运行我们训练好的实例来初步使用Lumos
我们为您提供了Cifar10数据集在Lenet5模型上的分类案例
请您先下载我们训练完成的权重文件
[<img src="https://img.shields.io/badge/Lumos-W-brightgreen" />]
请将下载的权重文件存放至Lumos根目录下，并使用如下命令运行

```shell
$ ./lumos.exe gpu
```
您将会看到如下输出
```

```
最后一行您将看到测试数据集的正确率
更为详细的Lumos使用教程请参考[使用手册]




## 联系我们：

####     邮箱：yuzhuoandlb@foxmail.com

​    