#include "softmax.h"

void softmax(float *data, int num, float *space)
{
    float M = 0;
    float res = 0;
    max_cpu(data, num, &M);
    for (int i = 0; i < num; ++i){
        res += exp(data[i] - M);
    }
    for (int i = 0; i < num; ++i){
        space[i] = exp(data[i] - M) / res;
    }
}

void softmax_grident(float *data, int num, float *space)
{
    float M = 0;
    float res = 0;
    max_cpu(data, num, &M);
    for (int i = 0; i < num; ++i){
        res += exp(data[i] - M);
    }
    for (int i = 0; i < num; ++i){
        float x = exp(data[i] - M);
        space[i] = (res + x) * x;
    }
}
