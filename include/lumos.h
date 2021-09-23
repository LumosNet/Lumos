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

typedef tensor array, Array, victor, Victor, image, Image, matrix, Matrix;

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
    CONVOLUTIONAL, POOLING, ACTIVE, FULLCONNECT
} LayerType;

typedef enum {
    MAX, AVG
} PoolingType;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

typedef struct layer{
    LayerType type;
    PoolingType pool;
    int *index;
    Tensor **input;
    Tensor **output;

    int width;
    int height;

    int filters;
    int ksize;
    int stride;
    int pad;

    Array **weights;

    void (*forward)  (struct layer , struct network);
    void (*backward) (struct layer , struct network);
    Activate active;
    Gradient gradient;
} layer, Layer;

typedef struct network{
    int n;
    int batch;
    int width;
    int height;
    float learning_rate;
    Layer **layers;
} network, Network, NetWork;

#ifdef __cplusplus
}
#endif

#endif