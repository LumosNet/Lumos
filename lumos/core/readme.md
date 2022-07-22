# CORE

Lumos框架核心组件，有如下子模块

- ops
- cu_ops
- tool
- layer
- graph
- session

模块关系如下图：



## OPS

operations（OPS）算子，算子即基本运算模块，包含线性代数计算、数字图像算法和基本运算组件

当前，该模块包含如下内容

| 名     | 描述                               |
| ------ | ---------------------------------- |
| cpu    | 基于数组的基本操作和计算           |
| bias   | 偏执项相关计算                     |
| active | 激活函数相关计算                   |
| gemm   | 矩阵乘（带转置的矩阵乘）           |
| im2col | 图像的im2col操作（多用于卷积计算） |
| image  | 图像基本操作和计算                 |

该模块为lumos最基础组件，关于各子模块的详细信息，请参考OPS模块的自述文件



## CU_OPS

与OPS模块相对应，是OPS模块的CUDA实现

与GPU加速相关的内容集中于该模块中



## TOOL

暂略



## LAYER

由多种基础算子组成的网络基础层级结构

当前该模块包含如下内容

| 名                  | 描述                       |
| ------------------- | -------------------------- |
| layer               | 所有layer的基类            |
| connect_layer       | 全连接层                   |
| convolutional_layer | 卷积层                     |
| avgpool_layer       | 均值池化层                 |
| maxpool_layer       | 最大值池化层               |
| im2col              | 多维（三维）图像转一维数据 |
| mse                 | mse损失层                  |
| softmax             | softmax+交叉熵损失层       |

layer作为构建计算图graph的主要组件，关于每一类层的详细描述请参考layer目录下的自述文件



## GRAPH

计算图，用于构建网络的数据流和操作。计算图的概念来源于Tensorflow中计算图，在本框架中使用该概念构建的主要意义是增加其迁移和重复使用能力。





## SESSION

与计算图绑定，管理网络训练过程中的内存，并控制调度流程

