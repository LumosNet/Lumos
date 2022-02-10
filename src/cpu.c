#include "cpu.h"

void fill_cpu(float *data, int len, float x, int offset)
{
    for (int i = 0; i < len; i += offset){
        data[i] = x;
    }
}

void multy_cpu(float *data, int len, float x, int offset)
{
    for (int i = 0; i < len; i += offset){
        data[i] *= x;
    }
}

void add_cpu(float *data, int len, float x, int offset)
{
    for (int i = 0; i < len; i += offset){
        data[i] += x;
    }
}