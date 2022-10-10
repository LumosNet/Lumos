#ifndef RANDOM_H
#define RANDOM_H

#ifdef __cplusplus
extern "C"{
#endif

float uniform_data(float a, float b, int *seed);
float guass_data(float mean, float sigma, int *seed);

void guass_list(float mean, float sigma, int seed, int num, float *space);

#ifdef  __cplusplus
}
#endif

#endif