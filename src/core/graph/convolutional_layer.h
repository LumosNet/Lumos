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

Layer make_convolutional_layer(CFGParams *p);

void forward_convolutional_layer(Layer l);
void backward_convolutional_layer(Layer l, float *n_delta);

void update_convolutional_layer(Layer l, float rate, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif