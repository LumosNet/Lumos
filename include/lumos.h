#ifndef INCLUDE_H
#define INCLUDE_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 行优先存储 */
typedef struct tensor{
    int       dim;
    int       *size;
    int       num;
    float     *data;
} tensor, Tensor;

typedef tensor array, Array, Vector, Vector, image, Image, matrix, Matrix;

typedef float   (*activate)(float);
typedef float   (*gradient)(float);

typedef activate Activate;
typedef gradient Gradient;

typedef enum {
    STAIR,
    HARDTAN,
    LINEAR,
    LOGISTIC,
    LOGGY,
    RELU,
    ELU,
    SELU,
    RELIE,
    RAMP,
    LEAKY,
    TANH,
    PLSE,
    LHTAN
} Activation;

typedef enum {
    CONVOLUTIONAL, POOLING, ACTIVATION, CONNECT, SOFTMAX
} LayerType;

typedef enum {
    MAX, AVG
} PoolingType;

typedef enum {
    MSE, MAE, HUBER, QUANTILE, CROSS_ENTROPY, HINGE
} LossType;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

typedef void (*forward)  (struct layer * , struct network *);
typedef void (*backward) (struct layer * , struct network *);

typedef forward Forward;
typedef backward Backward;

typedef void (*saveweight) (struct layer *, FILE *);
typedef void (*loadweight) (struct layer *, FILE *);

typedef saveweight SaveWeight;
typedef loadweight LoadWeight;

typedef struct layer{
    LayerType type;
    PoolingType pool;
    LossType loss;
    int *index;
    int input_h;
    int input_w;
    int input_c;
    int output_h;
    int output_w;
    int output_c;
    Tensor **input;
    Tensor **output;

    int filters;
    int ksize;
    int stride;
    int pad;
    int group;

    int bias;
    int batchnorm;

    // 损失计算参数
    float theta;
    float gamma;

    Array *kernel_weights;
    Array *bias_weights;

    Forward forward;
    Backward backward;

    Activate active;
    Gradient gradient;

    SaveWeight sweights;
    LoadWeight lweights;
} layer, Layer;

typedef struct network{
    int n;
    int batch;
    int width;
    int height;
    int channel;
    float learning_rate;
    Tensor **delta;
    Layer **layers;

    Tensor **labels;
} network, Network, NetWork;

#ifdef __cplusplus
}
#endif

#endif