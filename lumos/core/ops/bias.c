#include "bias.h"

void add_bias(float *origin, float *bias, int n, int size)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            origin[i * size + j] += bias[i];
        }
    }
}

void scale_bias(float *origin, float *bias, int n, int size)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            origin[i * size + j] *= bias[i];
        }
    }
}
