---
layout: default
permalink: /lumos/Install/
---
# Lumos

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

当前Lumos无法在除Linux外的其他平台运行，如果您希望在您的Windows上运行，您可以借助wsl2来进行。这是windows推出的Linux Subsystem，并且wsl2支持共享主机GPU，相关安装方法请参考[wsl文档](https://docs.microsoft.com/zh-cn/windows/wsl/)，如果您成功安装wsl2，那么请参考上文Linux下Lumos安装方法即可成功安装Lumos
