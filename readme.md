# Lumos

### 简介

Lumos是一个以个人学习为目的而开发的深度学习框架，以darknet框架作为主要参考。Lumos框架以C语言为主要开发语言，gcc作为主要编译器，当然该框架也可以轻松的在Windows上进行开发。

Lumos一词来源《哈利波特》中的咒语荧光闪烁，该框架旨在提供一个简洁的轻量级深度学习框架，帮助学习、研究深度学习算法和底层实现的学者和爱好者更容易学习相关内容，希望该框架是你学习过程中指引你的荧光。

Lumos的目标从来不是为了比肩TensorFlow或者Pytorch这样的顶级开源框架，而是希望更好的展现底层算法实现，提供给使用者更灵活的使用体验。同时希望有更多的人不止能在上层框架构建成熟应用，更能对底层算法原理和实现技巧有更多的关注。

### 开发状况

当前Lumos提供基础的全连接神经网络和卷积神经网络基本组件，使用配置文件的方式提供快速构建神经网络的方法。如果你有Darknet框架的使用经验，那么你可以快速上手Lumos。

Lumos框架现在只提供了最基本的网络组件，对于一些特定算法的处理方法会在后续开发继续完善。

当前该框架还未能提供GPU加速，你所构建的所有内容都将在CPU上进行计算。

### 示例

以下将演示如何在Lumos上构建一个简单的神经网络。

我们所熟知的大部分机器学习算法解决的都是线性问题，对于非线性问题的建模在早期人工智能发展中是一个难以解决的问题。而神经网络成功解决非线性建模问题，是人工智能发展的一大突破。那么以下示例就用神经网络构建一个典型的非线性模型，用神经网络解决XOR（异或）问题。

同或：相同为真，不同为假

异或：相同为假，不同为真

|      | 同或 | 异或 | 与 | 或 |
| ---- | ---- | ---- | -- | -- |
| 0，1 | 0    | 1    | 0  | 1  |
| 1，0 | 0    | 1    | 0  | 1  |
| 0，0 | 1    | 0    | 0  | 0  |
| 1，1 | 1    | 0    | 1  | 1  |

![](https://gitee.com/lumos-net/lumos/raw/master/img/1035701-20170414195906923-1457391618.png)

如上图所示，与、或等模型都是线性可分的，只有异或问题线性不可分，这就是XOR问题的非线性。

#### 构建网络结构

该实例采用的神经网络结构如下图

![](https://gitee.com/lumos-net/lumos/raw/master/img/XOR网络.drawio.png)

输入为一个一行两列的行向量：（1，1），（1，0），（0，1），（0，0）

两层全连接层，第一层有两个节点，第二层有一个节点

最后一层为损失层，采用MSE（均方差）损失。

Lumos框架采用配置文件的方式构建网络结构，需要编写一个cfg格式文件，如下：

```json
[XOR]
batch=2
width=2
height=1
channel=1

learning_rate=1

[im2col]
flag=1

[connect]
output=2
bias=1
active=logistic

[connect]
output=1
bias=1
active=logistic

[mse]
group=1
noloss=0

```

如上所示，我们已经构建好神经网络的结构。特别说明一下其中一些配置参数。

**说明**

Lumos框架主要服务与卷积神经网络，所有的数据都要以图像数据的形式传入，所以需要填写输入数据的大小（行、列、通道数）

由于输入为图片形式，是一个三维数据，直接对接全连接层是不行的，需要将三维数据转为二维向量，Lumos提供了一个im2col层来实现

**描述**

**[XOR]**

我们的输入是两个值组成的行向量，所以输入数据的大小为（行：1、列：2、通道数：1）

batch为批次梯度下降中每一批次的数量，learning_rate为学习率

**[im2col]**

flag=1代表将三维数据转为行向量

**[connect]**

output表示该层的输出大小，也就是该层的节点数

bias表示是否计算偏移量，bias=1表示需要计算偏移量

active表示该层的激活函数，这里采用logistic函数

**[mes]**

group代表类别标签的位数，XOR问题是一个二分类问题，输出0代表第一类，输出1代表第二类，类别标签0、1的位数是1

noloss表示该层在反向推导过程中是否需要合并上层传递的梯度

#### 数据准备

你需要准备三个文件

数据路径目录文件：

```
/home/btboay/lumos-matrix/XOR/data/0_1.png
/home/btboay/lumos-matrix/XOR/data/1_1.png
/home/btboay/lumos-matrix/XOR/data/0_2.png
/home/btboay/lumos-matrix/XOR/data/1_2.png
```

存放所有训练数据的路径

标签路径目录文件：

```
/home/btboay/lumos-matrix/XOR/data/0_1.txt
/home/btboay/lumos-matrix/XOR/data/1_1.txt
/home/btboay/lumos-matrix/XOR/data/0_2.txt
/home/btboay/lumos-matrix/XOR/data/1_2.txt
```

存放所有训练数据的标签文件路径

标签文件内容如下：

```
0
```

记录该数据的标签，该实例标签只有0和1

数据总览文件：

```
classes=2
data=/home/btboay/lumos-matrix/XOR/data.txt
label=/home/btboay/lumos-matrix/XOR/label.txt
```

- classes：类别数量
- data：数据路径目录文件路径（上述第一个文件）
- label：标签路径目录文件路径（上述第二个文件）

由于输入数据必须是图像数据，我已经准备好了对应的数据文件以及所有配置文件：

链接：https://pan.baidu.com/s/1QwmyTUudDSqeJOZCMrXTJg
提取码：0cgm

#### 训练

当你照着以上步骤一直到这里时，你离成功运行你的模型已经只有一步之遥了。

现在你需要创建一个主函数，并调用Lumos准备好的API来启动你的模型，代码如下：

```c
#include <stdio.h>
#include <stdlib.h>

#include "lumos.h"
#include "network.h"

int main(int argc, char **argv)
{
    Network *net = load_network("./cfg/xor.cfg");
    init_network(net, "./XOR/xor.data", NULL);
    train(net, 200);
    return 0;
}
```

你应该至少包含lumos.h和network.h两个头文件

分别调用load_network和init_network加载并初始化神经网络，最后调用train开始训练神经网络

训练结束后，权重文件会自动保存到./data/w.weights中

#### 测试

当前测试过程还在优化，虽然提供了一个test接口但是并不好用
