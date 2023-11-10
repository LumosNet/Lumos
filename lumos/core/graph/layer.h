#ifndef LAYER_H
#define LAYER_H

#include <stdlib.h>
#include <stdio.h>

#include "active.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct layer Layer;

typedef void (*init)     (struct layer*, int, int, int);
typedef void (*release)  (struct layer*);
typedef void (*forward)  (struct layer, int);
typedef void (*backward) (struct layer, float, int, float*);
typedef void (*update)   (struct layer, float, int, float*);

typedef init     Init;
typedef release  Release;
typedef forward  Forward;
typedef backward Backward;
typedef update   Update;

struct layer{
    char *ID;
    int input_h;
    int input_w;
    int input_c;
    int output_h;
    int output_w;
    int output_c;

    int inputs;
    int outputs;
    int workspace_size;

    float *input;
    float *output;
    float *delta;
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

    int bias;
    int batchnorm;
    int train;

    int index;

/* Dropout相关 */
    float probability;

    Layer *shortcut;
    int shortcut_index;

    float *kernel_weights;
    float *bias_weights;
    float *normalize_weights;

    float *update_kernel_weights;
    float *update_bias_weights;
    float *update_normalize_weights;

    Init     init;
    Release  release;
    Forward  forward;
    Backward backward;
    Update   update;

    Activate active;
    Gradient gradient;
};

#ifdef __cplusplus
}
#endif

#endif