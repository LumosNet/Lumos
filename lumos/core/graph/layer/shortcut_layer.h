#ifndef SHORTCUT_LAYER_H
#define SHORTCUT_LAYER_H

#include <stdlib.h>
#include <stdio.h>

#include "layer.h"
#include "cpu.h"
#include "shortcut.h"

#include "shortcut_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_shortcut_layer(int index, char *active);
void init_shortcut_layer(Layer *l, int w, int h, int c, Layer *shortcut);

void forward_shortcut_layer(Layer l, int num);
void backward_shortcut_layer(Layer l, float rate, int num);

#ifdef __cplusplus
}
#endif

#endif
