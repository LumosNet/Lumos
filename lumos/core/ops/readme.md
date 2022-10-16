# OPS

ops(operations 算子)模块包括所有基础计算模块，无论是简单的加减还是复杂的数字图像处理，这些单纯的数值计算算法都在该模块下实现。Layer模块对该模块强依赖，Layer模块依赖各类算子实现不同的计算图过程。

ops模块包含如下子类（在C语言中通常将一个文件视为一个类）：

| 名     | 描述               |
| ------ | ------------------ |
| active | 激活函数           |
| bias   | 偏执项计算         |
| cpu    | Host侧数组相关计算 |
| gemm   | 矩阵乘             |
| im2col | im2col算法         |
| image  | 数字图像处理       |

下面将对各子类进行详细说明



## active

该类实现所有激活函数以及各函数的梯度计算，包含如下函数：

| 激活函数          | 激活函数梯度计算  |
| ----------------- | ----------------- |
| stair_activate    | stair_gradient    |
| hardtan_activate  | hardtan_gradient  |
| linear_activate   | linear_gradient   |
| logistic_activate | logistic_gradient |
| loggy_activate    | loggy_gradient    |
| relu_activate     | relu_gradient     |
| elu_activate      | elu_gradient      |
| selu_activate     | selu_gradient     |
| relie_activate    | relie_gradient    |
| ramp_activate     | ramp_gradient     |
| leaky_activate    | leaky_gradient    |
| tanh_activate     | tanh_gradient     |
| plse_activate     | plse_gradient     |
| lhtan_activate    | lhtan_gradient    |

我们采用枚举类型枚举出所有激活函数，并使用它来表示您采用的激活类型：

| enum     |
| -------- |
| STAIR    |
| HARDTAN  |
| LINEAR   |
| LOGISTIC |
| LOGGY    |
| RELU     |
| ELU      |
| SELU     |
| RELIE    |
| RAMP     |
| LEAKY    |
| TANH     |
| PLSE     |
| LHTAN    |

通常我们使用字符串来表明所希望选择的激活函数类型，*load_activate_type*函数提供了该功能

```c
Activation load_activate_type(char *activate);
```

该函数通过输入的字符串获取对应的激活函数类型，例如：

```c
active_type = load_activate_type("logistic");
```

则*active_type*为LOGISTIC

但是最终我们希望获取到对于的激活函数或梯度计算函数，而*load_activate*与*load_gradient*提供该功能：

```c
Activate load_activate(Activation TYPE);
```

```c
Gradient load_gradient(Activation TYPE);
```

只需提供正确的激活函数类型，则会返回对应激活函数或梯度计算函数对象

最终这些激活函数或梯度计算函数将用于处理连续存储空间的数据（数组），为了方便我们提供了相关的功能函数*activate_list*与*gradient_list*

```c
void activate_list(float *origin, int num, Activate a);
```

```c
void gradient_list(float *origin, int num, Gradient g);
```

- origin：需要处理的线性空间
- num：线性空间中存放的数据数量
- a/g：激活/梯度计算函数

示例：

```c
float a[10] = {-5, -4, -3, -2, -1, 1, 2, 3, 4, 5};
Activation type = load_activate_type("relu");
Activate active = load_activate(type);
activate_list(a, 10, active);
```

最终数组a的值为：

```
0, 0, 0, 0, 0, 1, 2, 3, 4, 5
```



## bias

该模块处理网络层的bias（偏置项），当前该模块只有一个方法*add_bias*，用于在计算中加入偏置项

```c
void add_bias(float *origin, float *bias, int n, int size)
{
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < size; ++j){
            origin[i*size + j] += bias[i];
        }
    }
}
```

当前网络中只有connect_layer和convolutional_layer中涉及偏置项计算，由于卷积网络和全连接网络的bias计算不一样，所以采用双层循环以满足对同一channel数据累加同一个bias值





## cpu

该模块包含所有数组计算方法，例如对数组中每一个数据累加一个值，等类似操作

| 名               | 描述                                                 |
| ---------------- | ---------------------------------------------------- |
| fill_cpu         | 向数组中填充指定数据                                 |
| multy_cpu        | 对数组中每一个元素乘一个相同值                       |
| add_cpu          | 对数组中每一个元素加一个相同值                       |
| min_cpu          | 获取数组中的最小值                                   |
| max_cpu          | 获取数组中的最大值                                   |
| sum_cpu          | 计算数组数据的累加值                                 |
| mean_cpu         | 计算数组数据的均值                                   |
| one_hot_encoding | 独热编码                                             |
| add              | 两个数组对应元素相加                                 |
| subtract_cpu         | 两个数组对应元素相减                                 |
| multiply_cpu         | 两个数组对应元素相乘                                 |
| divide_cpu           | 两个数组对应元素相除                                 |
| saxpy_cpu            | 一个数组各元素乘一个相同值再与另一个数组元素对应相加 |





## gemm

通用矩阵乘模块，包含如下方法

| 名      | 说明                     |
| ------- | ------------------------ |
| gemm_nn | 矩阵乘，两个矩阵不做转置 |
| gemm_tn | 矩阵乘，第一个矩阵做转置 |
| gemm_nt | 矩阵乘，第二个矩阵做转置 |
| gemm_tt | 矩阵乘，两个矩阵都做转置 |
|         |                          |
| gemm    | 调用矩阵乘运算的通用接口 |

```c
void gemm(int TA, int TB, int AM, int AN, int BM, int BN, float ALPHA, 
        float *A, float *B, float *C);
```

- TA：第一个矩阵是否转置，1是，0否
- TB：第二个矩阵是否转置，1是，0否
- AM：第一个矩阵的行数
- AN：第一个矩阵的列数
- BM：第二个矩阵的行数
- BN：第二个矩阵的列数
- ALPHA：允许您对计算结果进行缩放，我们会对所有结果数据乘ALPHA
- A：第一个矩阵
- B：第二个矩阵
- C：计算结果的存储空间

示例：

```c
float a[3*3] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
float b[3*3] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
float c[3*3];
gemm(0, 0, 3, 3, 3, 3, 1, a, b, c);
for (int i = 0; i < 3; ++i){
    for (int j = 0; j < 3; ++j){
        printf("%f ", c[i*3+j]);
    }
    printf("\n");
}
```

结果为

```
30  36  42
66  81  96
102 126 150
```





## im2col

该模块实现卷积运算中的im2col相关方法，在这里不过多介绍

| 名               | 描述                                               |
| ---------------- | -------------------------------------------------- |
| im2col           | 多维数据转一维数据，使卷积过程直接采用矩阵计算实现 |
| col2im           | 一维数据转换会多维数据                             |
| im2col_get_pixel | 特殊的元素索引算法                                 |





## image

数字图像相关处理模块

| 名              | 描述         |
| --------------- | ------------ |
| load_image_data | 读取图像数据 |
| save_image_data | 保存图像数据 |
| resize_im       | 转变图像大小 |

resize_im采用双线性插值算法缩放图像大小



