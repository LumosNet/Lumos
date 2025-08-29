#ifndef LAYER_H
#define LAYER_H

#include <stdlib.h>
#include <stdio.h>

#include "active.h"
#include "active_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CPU 0
#define GPU 1

typedef enum {
    CONVOLUTIONAL, CONNECT, IM2COL, MAXPOOL, AVGPOOL, \
    DROPOUT, SOFTMAX, SHORTCUT, NORMALIZE, \
    MSE, MAE, CE
} LayerType;

typedef enum {
    CONSTANT_I, NORMAL_I, UNIFORM_I, KAIMING_NORMAL_I, KAIMING_UNIFORM_I
} InitType;

typedef struct initcpt{
    InitType initype;
    float x;
    float mean;
    float std;
    float a;
    float min;
    float max;
    char *mode;
    char *nonlinearity;
} initcpt, InitCpt;

typedef struct initcpt InitCpt;
typedef struct layer Layer;

typedef void (*forward)  (struct layer, int);
typedef void (*backward) (struct layer, float, int, float*);
typedef void (*update) (struct layer);
typedef forward Forward;
typedef backward Backward;
typedef update Update;

typedef void (*forward_gpu)  (struct layer, int);
typedef void (*backward_gpu) (struct layer, float, int, float*);
typedef void (*update_gpu) (struct layer);
typedef forward_gpu ForwardGpu;
typedef backward_gpu BackwardGpu;
typedef update_gpu UpdateGpu;

typedef void (*initialize) (struct layer *, int, int, int, int);
typedef void (*initialize_gpu) (struct layer *, int, int, int, int);
typedef initialize Initialize;
typedef initialize_gpu InitializeGpu;

typedef void (*weightinit) (struct layer, FILE*);
typedef weightinit WeightInit;
typedef void (*weightinit_gpu) (struct layer, FILE*);
typedef weightinit_gpu WeightInitGpu;

typedef void (*saveweights) (struct layer, FILE*);
typedef saveweights SaveWeights;
typedef void (*saveweights_gpu) (struct layer, FILE*);
typedef saveweights_gpu SaveWeightsGpu;

typedef void (*free_layer) (struct layer);
typedef free_layer FreeLayer;
typedef void (*free_layer_gpu) (struct layer);
typedef free_layer_gpu FreeLayerGpu;

struct layer{
    LayerType type;
    int status;
    int input_h;
    int input_w;
    int input_c;
    int output_h;
    int output_w;
    int output_c;

    int inputs;
    int outputs;

    int workspace_size;
    int truth_num;

    float *input;
    float *output;
    float *delta;
    float *truth;
    float *loss;
    float *workspace;

    int *maxpool_index;
    //为社么是指针
    int *dropout_rand;

    int filters;
    int ksize;
    int stride;
    int pad;
    int group;

    int bias;
    int normalize;
    // dropout 占比
    float probability;

    Layer *shortcut;
    int shortcut_index;

    float *kernel_weights;
    float *bias_weights;

    float *update_kernel_weights;
    float *update_bias_weights;

    float *mean;
    float *variance;
    float *rolling_mean;
    float *rolling_variance;
    float *x_norm;
    float *normalize_x;
    float *mean_delta;
    float *variance_delta;

    float *bn_scale;
    float *bn_bias;
    float *update_bn_scale;
    float *update_bn_bias;

    Forward forward;
    Backward backward;
    Update update;

    ForwardGpu forwardgpu;
    BackwardGpu backwardgpu;
    UpdateGpu updategpu;

    Initialize initialize;
    InitializeGpu initializegpu;

    WeightInit weightinit;
    WeightInitGpu weightinitgpu;

    Activation active;
    SaveWeights saveweights;
    SaveWeightsGpu saveweightsgpu;

    FreeLayer freelayer;
    FreeLayerGpu freelayergpu;

    InitCpt *initcpt;
};

#ifdef __cplusplus
}
#endif

#endif