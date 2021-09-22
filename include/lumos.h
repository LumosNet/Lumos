#ifndef INCLUDE_H
#define INCLUDE_H

#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "array.h"
#include "victor.h"
#include "list.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    STAIR, HARDTAN, LINEAR, LOGISTIC, LOGGY, RELU, \
    ELU, SELU, RELIE, RAMP, LEAKY, TANH, PLSE, LHTAN
} activation, Activation;

typedef enum {
    CONVOLUTIONAL, MAXPOOL, AVGPOOL, ACTIVE
} layer_type, LayerType;

typedef struct network{
    int n;
    int batch;
    Layer **layers;
} network, Network, NetWork;

typedef struct layer{
    LayerType type;
    Activation activation;
    Tensor *input;
    Tensor *output;

    int ksize;
    int n;
    int stride;
    int pad;

    Array **weights;

    void (*forward)  (struct layer , struct network);
    void (*backward) (struct layer , struct network);
} layer, Layer;

#ifdef __cplusplus
}
#endif

#endif