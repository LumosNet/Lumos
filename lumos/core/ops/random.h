#ifndef RANDOM_H
#define RANDOM_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifdef __cplusplus
extern "C"{
#endif

#define TWO_PI 6.2831853071795864769252866f

float uniform_data(float a, float b, int *seed);
float guass_data(float mean, float sigma, int *seed);

void uniform_list(float a, float b, int num, float *space);
void guass_list(float mean, float sigma, int seed, int num, float *space);
void normal_list(int num, float *space);
void uniform_int_list(int a, int b, int num, float *space);

float rand_normal();
float rand_uniform(float min, float max);

float rand_normal();

#ifdef  __cplusplus
}
#endif

#endif