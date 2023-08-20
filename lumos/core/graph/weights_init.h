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
    char *mode;
};

void val_init(Layer *l, float val);
void uniform_init(Layer *l, float mean, float variance);
void normal_init(Layer *l, float mean, float variance);
void xavier_uniform_init(Layer *l);
void xavier_normal_init(Layer *l);
void kaiming_uniform_init(Layer *l, char *mode);
void kaiming_normal_init(Layer *l, char *mode);
void he_init(Layer *l);

void connect_layer_w_init(Layer *l);
void convolutional_layer_w_init(Layer *l);

Initializer val_initializer(float val);
Initializer uniform_initializer(float mean, float variance);
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
