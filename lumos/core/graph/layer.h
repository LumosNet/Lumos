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
    VALFILL, UNIFORM, NORMAL, XAVIERU, XAVIERN,, KAIMINGU, KAIMINGN, HE
} InitType;

typedef enum {
    CONVOLUTIONAL, ACTIVATION, CONNECT, IM2COL, MAXPOOL, AVGPOOL, \
    DROPOUT, MSE, SOFTMAX, SHORTCUT, NORMALIZE
} LayerType;

typedef struct layer Layer;

typedef void (*initialize) (struct layer, int, int, int);
typedef void (*forward)  (struct layer, int);
typedef void (*backward) (struct layer, float, int, float*);
typedef initialize Initialize;
typedef forward Forward;
typedef backward Backward;

typedef void (*initialize_gpu) (struct layer, int, int, int);
typedef void (*forward_gpu)  (struct layer, int);
typedef void (*backward_gpu) (struct layer, float, int, float*);
typedef initialize_gpu Initialize_Gpu;
typedef forward_gpu Forward_Gpu;
typedef backward_gpu Backward_Gpu;

typedef int (*get_float_calculate_times) (struct layer*);
typedef get_float_calculate_times GetFloatCalculateTimes;

struct layer{
    int subdivision;
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
    int normalize_weights_size;
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
    float *normalize_weights;

    float *update_kernel_weights;
    float *update_bias_weights;
    float *update_normalize_weights;

    int mean_size;
    int variance_size;

    float *mean;
    float *variance;
    float *rolling_mean;
    float *rolling_variance;
    float *x_norm;
    float *normalize_x;

    Initialize initialize;
    Forward forward;
    Backward backward;

    Initialize_Gpu initialize_gpu;
    Forward_Gpu forward_gpu;
    Backward_Gpu backward_gpu;

    Activation active;
    Activation gradient;

    GetFloatCalculateTimes get_fct;
};

#ifdef __cplusplus
}
#endif

#endif