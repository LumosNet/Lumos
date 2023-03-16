#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cpu.h"
#include "random.h"

#ifdef __cplusplus
extern "C"{
#endif

void uniform_init(int seed, float a, float b, int num, float *space);
void guass_init(int seed, float mean, float variance, int num, float *space);
void xavier_init(int seed, int inp, int out, float *space);
void kaiming_init(int seed, int inp, int out, float *space);

void xavier_uniform(int seed, int inp, int out, float *space);
void xavier_normal(int seed, int inp, int out, float *space);

#ifdef  __cplusplus
}
#endif

#endif