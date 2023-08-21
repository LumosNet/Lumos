#ifndef WEIGHTS_INIT_H
#define WEIGHTS_INIT_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "layer.h"
#include "cpu.h"
#include "random.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct initializer Initializer;

struct initializer{
    char *type;
    float val;
    float mean;
    float variance;
    float l;
    float r;
    char *mode;
};

void connect_layer_weights_init(Layer *l, Initializer init);
void convolutional_layer_weights_init(Layer *l, Initializer init);

void val_init(float *space, int num, float val);
void uniform_init(float *space, int num, float l, float r);
void normal_init(float *space, int num, float mean, float variance);
void xavier_uniform_init(float *space, int num, float x);
void xavier_normal_init(float *space, int num, float x);
void kaiming_uniform_init(char *mode, float *space, int num, int inp, int out);
void kaiming_normal_init(char *mode, float *space, int num, int inp, int out);

Initializer val_initializer(float val);
Initializer uniform_initializer(float l, float r);
Initializer normal_initializer(float mean, float variance);
Initializer xavier_uniform_initializer();
Initializer xavier_normal_initializer();
Initializer kaiming_uniform_initializer(char *mode);
Initializer kaiming_normal_initializer(char *mode);
Initializer he_initializer();

#ifdef __cplusplus
}
#endif

#endif
