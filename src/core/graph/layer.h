#ifndef LAYER_H
#define LAYER_H

#include <stdlib.h>
#include <stdio.h>

#include "avgpool_layer.h"
#include "connect_layer.h"
#include "convolutional_layer.h"
#include "im2col_layer.h"
#include "maxpool_layer.h"
#include "mse_layer.h"
#include "softmax_layer.h"
#include "cfg_f.h"

#ifdef __cplusplus
extern "C" {
#endif

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

    int inputs;
    int outputs;
    int kernel_weights_size;
    int bias_weights_size;
    int deltas;
    int workspace_size;

    char *active_str;

    float *input;
    float *output;
    float *delta;

    float *workspace;

    int *maxpool_index;

    int inputs;
    int outputs;

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

    float *kernel_weights;
    float *bias_weights;

    float *update_kernel_weights;
    float *update_bias_weights;

    Forward forward;
    Backward backward;

    Activate active;
    Gradient gradient;

    Update update;
} layer, Layer;

typedef float   (*activate)(float);
typedef float   (*gradient)(float);

typedef activate Activate;
typedef gradient Gradient;

typedef void (*forward)  (struct layer);
typedef void (*backward) (struct layer, float*);
typedef forward Forward;
typedef backward Backward;

typedef void (*update) (struct layer, float, float*);
typedef update Update;

Layer create_layer(CFGPiece *p);

#ifdef __cplusplus
}
#endif

#endif