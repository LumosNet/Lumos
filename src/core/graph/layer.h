#ifndef LAYER_H
#define LAYER_H

#include <stdlib.h>
#include <stdio.h>

#include "active.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CONVOLUTIONAL, ACTIVATION, CONNECT, IM2COL, MAXPOOL, AVGPOOL
} LayerType;

typedef struct layer Layer;

typedef void (*forward)  (struct layer, int);
typedef void (*backward) (struct layer, int, float*);
typedef forward Forward;
typedef backward Backward;

typedef void (*update) (struct layer, float, float*);
typedef update Update;

struct layer{
    LayerType type;
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

    char *active_str;

    float *input;
    float *output;
    float *label;
    float *delta;

    float *workspace;

    int *maxpool_index;

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

    float *kernel_weights;
    float *bias_weights;

    float *update_kernel_weights;
    float *update_bias_weights;

    Forward forward;
    Backward backward;

    Activate active;
    Gradient gradient;

    Update update;
};


#ifdef __cplusplus
}
#endif

#endif