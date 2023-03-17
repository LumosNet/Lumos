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
    float scale;
};

void val_init(Layer *l, float val, float scale);
void uniform_init(Layer *l, float mean, float variance, float scale);
void normal_init(Layer *l, float mean, float variance, float scale);
void xavier_uniform_init(Layer *l, float scale);
void xavier_normal_init(Layer *l, float scale);
void kaiming_uniform_init(Layer *l, float scale, char *mode);
void kaiming_normal_init(Layer *l, float scale, char *mode);

Initializer val_initializer(float val, float scale);
Initializer uniform_initializer(float mean, float variance, float scale);
Initializer normal_initializer(float mean, float variance, float scale);
Initializer xavier_uniform_initializer(float scale);
Initializer xavier_normal_initializer(float scale);
Initializer kaiming_uniform_initializer(float scale, char *mode);
Initializer kaiming_normal_initializer(float scale, char *mode);

#ifdef __cplusplus
}
#endif

#endif
