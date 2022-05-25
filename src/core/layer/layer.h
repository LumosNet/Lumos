#ifndef LAYER_H
#define LAYER_H

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float   (*activate)(float);
typedef float   (*gradient)(float);

typedef activate Activate;
typedef gradient Gradient;

struct layer;
typedef struct layer layer;

typedef void (*forward)  (struct layer, struct network);
typedef void (*backward) (struct layer, struct network);

typedef forward Forward;
typedef backward Backward;

typedef void (*saveweight) (struct layer, FILE *);
typedef void (*loadweight) (struct layer, FILE *);

typedef saveweight SaveWeight;
typedef loadweight LoadWeight;

typedef void (*update) (struct layer, struct network);
typedef update Update;

typedef enum {
    CONVOLUTIONAL, ACTIVATION, CONNECT, IM2COL, MSE, SOFTMAX, \
    MAXPOOL, AVGPOOL
} LayerType;

typedef struct layer{
    LayerType type;
    int input_h;
    int input_w;
    int input_c;
    int output_h;
    int output_w;
    int output_c;
    size_t workspace_size;
    float *input;
    float *output;
    float *delta;

    int inputs;
    int outputs;

    int filters;
    int ksize;
    int stride;
    int pad;
    int group;

    int bias;
    int batchnorm;

    // 在网中的位置，0开始
    int i;

    // 损失计算参数
    int noloss;
    float *loss;
    float *truth;
    float theta;
    float gamma;

    // label标签信息
    char **labels;
    int labels_num;

    float *kernel_weights;
    float *bias_weights;

    Forward forward;
    Backward backward;

    Activate active;
    Gradient gradient;

    SaveWeight sweights;
    LoadWeight lweights;

    Update update;
} layer, Layer;

// 单独构建一个模块是不是更好
typedef struct Session{
    int epoch;
    int batch;
    float learning_rate;
    size_t workspace_size;
    float *workspace;
} Session;

Layer create_layer(Network *net, LayerParams *p, int h, int w, int c);

#ifdef __cplusplus
}
#endif

#endif