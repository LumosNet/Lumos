<div align="center">
  <img src="https://github.com/LumosNet/Lumos/blob/master/img/Lumos.png">
</div>

# Lumos

## 简介

[Lumos](https://gitee.com/lumos-net/lumos)是一个简洁的轻量级深度学习框架，帮助学习、研究深度学习算法和底层实现的学者和爱好者更容易学习相关内容，希望该框架是你学习过程中指引你的荧光。

Lumos的目标从来不是为了比肩TensorFlow或者Pytorch这样的顶级开源框架，而是希望更好的展现底层算法实现，提供给使用者更灵活的使用体验。同时希望有更多的人不止能在上层框架构建成熟应用，更能对底层算法原理和实现技巧有更多的关注。



## 开发状况

当前Lumos提供基础的全连接神经网络和卷积神经网络基本组件，可以使用我们所提供的API快速开发网络模型。

Lumos框架现在只提供了最基本的网络组件，对于一些特定算法的处理方法会在后续开发继续完善。

我们支持您使用GPU进行模型加速，我们使用cuda优化了相关算法，您可以轻松的调用GPU加速您的网络模型。



## 安装

### 下载安装包
版本                    | 链接                                                                                                                                                                           | Coretype
----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------
**v0.1**                 | [<img src="https://img.shields.io/badge/Lumos-CPU-brightgreen" />](https://github.com/LumosNet/Lumos-Build/raw/main/v0.1.0/lumos_0.1.0_linux_cpu.run)           | CPU
**v0.1**                 | [<img src="https://img.shields.io/badge/Lumos-GPU-brightgreen" />](https://github.com/LumosNet/Lumos-Build/raw/main/v0.1.0/lumos_0.1.0_linux_gpu.run)           | GPU



### **Linux**

使用如下命令进行安装

```shell
$ sudo sh lumos_0.1.0_linux_cpu.run
```

添加环境变量，在用户目录下~/.bashrc文件末尾添加如下语句

```
export PATH=/usr/local/lumos/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/lumos/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

添加完成后，在命令行运行

```shell
source ~/.bashrc
```

来激活相关设置，此时您已经完成lumos的安装

通过命令行验证安装

```shell
lumos --version
```

若出现以下版本信息，则您已经安装好了lumos

```shell
Lumos version: v0.1
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```



### **CUDA**

Lumos支持cuda加速，如果您拥有支持cuda的GPU，并且希望使用GPU加速您的算法，那么请提前安装cuda，相关安装方法请参考[cuda文档](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

安装文件分为CPU和GPU版，请酌情选择安装



## 运行

我们提供了丰富的用例demo，您可以尝试运行这些demo，来开启您的lumos之旅，请您从如下仓库clone我们的demos

```shell
git clone git@github.com:LumosNet/Lumos-Demos.git
```

您需要手动下载数据集，并放入demo目录下

数据集：

名                    | 链接                                                                                                                                                                           | 说明
----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------
**MNIST**                 | [<img src="https://img.shields.io/badge/Lumos-CPU-brightgreen" />](https://pan.baidu.com/s/1Qm7HRy0oVx-eiVl0jBxC5A?pwd=6bxh )           | 手写数字数据集
**XOR**                 | [<img src="https://img.shields.io/badge/Lumos-GPU-brightgreen" />](https://pan.baidu.com/s/1KMGSVsDKPFy7U9Wnfxd7yw?pwd=ec2o )           | 异或数据集

在Lumos-Demos根目录下新建data文件夹，并将您下载好的数据集放入data文件夹中
我们为您提供了编译的Makefile，您只需使用make命令编译即可

编译完成后使用如下指令查看相关可操作内容

```
./lumos.exe --help
```




### 使用
通过添加头文件: lumos.h
```
"lumos.h"
```
来调用我们提供的相关接口，接口详细信息请查看我们的接口描述文件
编译时请添加如下命令
链接动态库:
```shell
-llumos
```

头文件引用路径:
```shell
-I/usr/local/lumos/include/
```

编译时如果出现如下错误:
```shell
/usr/bin/ld: /usr/local/lumos/lib/../lib/liblumops.so: undefined reference to `omp_get_thread_num'
/usr/bin/ld: /usr/local/lumos/lib/../lib/liblumops.so: undefined reference to `omp_get_num_threads'
/usr/bin/ld: /usr/local/lumos/lib/../lib/liblumops.so: undefined reference to `GOMP_parallel'
collect2: error: ld returned 1 exit status
```

请添加如下编译参数:
```shell
-fopenmp
```

通常，正确的编译命令应该如下:
```shell
gcc -fopenmp main.c -I/usr/local/lumos/include/ -o main -llumos
```



## 联系我们：

####     邮箱：yuzhuoandlb@foxmail.com

​    