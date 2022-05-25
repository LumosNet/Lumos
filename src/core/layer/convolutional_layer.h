#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include <time.h>

#include "layer.h"
#include "cfg_f.h"
#include "image.h"
#include "active.h"
#include "bias.h"
#include "gemm.h"
#include "cpu.h"

#ifdef __cplusplus
extern "C"{
#endif

Layer make_convolutional_layer(CFGParams *p, int h, int w, int c);

void forward_convolutional_layer(Layer l, float *workspace);
void backward_convolutional_layer(Layer l, float *n_delta, float *workspace);

void update_convolutional_layer(Layer l, float rate, float *n_delta, float *workspace);

void save_convolutional_weights(Layer l, FILE *file);
void load_convolutional_weights(Layer l, FILE *file);

#ifdef __cplusplus
}
#endif

#endif