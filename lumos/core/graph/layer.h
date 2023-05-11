#ifndef LAYER_H
#define LAYER_H

#include <stdlib.h>
#include <stdio.h>

#include "active.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CPU 0
#define GPU 1

typedef enum {
    CONVOLUTIONAL, ACTIVATION, CONNECT, IM2COL, MAXPOOL, AVGPOOL, \
    DROPOUT, MSE, SOFTMAX, SHORTCUT, NORMALIZE
} LayerType;

typedef struct layer Layer;

typedef void (*forward)  (struct layer, int);
typedef void (*backward) (struct layer, float, int, float*);
typedef forward Forward;
typedef backward Backward;

typedef void (*update) (struct layer, float, int, float*);
typedef update Update;

typedef int (*get_float_calculate_times) (struct layer*);
typedef get_float_calculate_times GetFloatCalculateTimes;

struct layer{
    LayerType type;
    int coretype;
    int input_h;
    int input_w;
    int input_c;
    int output_h;
    int output_w;
    int output_c;

    int inputs;
    int outputs;
    int kernel_weights_size;
    int bias_weights_size;
    int deltas;
    int workspace_size;
    int label_num;

    char *active_str;

    float *input;
    float *output;
    float *delta;
    char **label;
    float *truth;
    float *loss;

    float *workspace;

    int *maxpool_index;
    int *dropout_rand;

    int filters;
    int ksize;
    int stride;
    int pad;
    int group;
    int im2col_flag;

    int weights;
    int bias;
    int batchnorm;

    // 在网中的位置，0开始
    int index;
    // 浮点数操作数
    int fops;
    // dropout 占比
    float probability;
    int train;

    Layer *shortcut;
    int shortcut_index;

    float *kernel_weights;
    float *bias_weights;

    float *update_kernel_weights;
    float *update_bias_weights;

    float *kernel_weights_gpu;
    float *bias_weights_gpu;

    float *update_kernel_weights_gpu;
    float *update_bias_weights_gpu;

    int mean_size;
    int variance_size;

    float *mean;
    float *variance;
    float *rolling_mean;
    float *rolling_variance;
    float *x_norm;
    float *normalize_x;

    Forward forward;
    Backward backward;

    Activation active;
    Activation gradient;

    Update update;
    GetFloatCalculateTimes get_fct;
};

#ifdef __cplusplus
}
#endif

#endif