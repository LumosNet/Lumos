#include "cpu.h"

void fill_cpu(float *data, int len, float x, int offset)
{
    for (int i = 0; i < len; i += offset)
    {
        data[i] = x;
    }
}

void multy_cpu(float *data, int len, float x, int offset)
{
    for (int i = 0; i < len; i += offset)
    {
        data[i] *= x;
    }
}

void add_cpu(float *data, int len, float x, int offset)
{
    for (int i = 0; i < len; i += offset)
    {
        data[i] += x;
    }
}

void min_cpu(float *data, int num, float *space)
{
    float min = data[0];
    for (int i = 1; i < num; ++i)
    {
        if (data[i] < min)
            min = data[i];
    }
    space[0] = min;
}

void max_cpu(float *data, int num, float *space)
{
    float max = data[0];
    for (int i = 1; i < num; ++i)
    {
        if (data[i] > max)
            max = data[i];
    }
    space[0] = max;
}

void sum_cpu(float *data, int num, float *space)
{
    float res = 0;
    for (int i = 0; i < num; ++i)
    {
        res += data[i];
    }
    space[0] = res;
}

void mean_cpu(float *data, int num, float *space)
{
    sum_cpu(data, num, space);
    space[0] /= (float)num;
}

void variance_cpu(float *data, float mean, int num, float *space)
{
    for (int i = 0; i < num; ++i){
        space[0] += powf((data[i] - mean), 2);
    }
    space[0] *= 1./num;
}

void matrix_add_cpu(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i)
    {
        space[i] = data_a[i] + data_b[i];
    }
}

void matrix_subtract_cpu(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i)
    {
        space[i] = data_a[i] - data_b[i];
    }
}

void matrix_multiply_cpu(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i)
    {
        space[i] = data_a[i] * data_b[i];
    }
}

void matrix_divide_cpu(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i)
    {
        space[i] = data_a[i] / data_b[i];
    }
}

void saxpy_cpu(float *data_a, float *data_b, int num, float x, float *space)
{
    for (int i = 0; i < num; ++i)
    {
        space[i] = data_a[i] + data_b[i] * x;
    }
}

void sum_channel_cpu(float *data, int h, int w, int c, float ALPHA, float *space)
{
    for (int k = 0; k < c; ++k){
        float sum = 0;
        for (int i = 0; i < h; ++i){
            for (int j = 0; j < w; ++j){
                int offset = k*h*w;
                sum += data[offset + i*w + j] * ALPHA;
            }
        }
        space[k] = sum;
    }
}

void one_hot_encoding(int n, int label, float *space)
{
    if (n == 1)
        space[0] = (float)label;
    else
    {
        for (int i = 0; i < n; ++i)
        {
            space[i] = (float)0;
        }
        space[label] = (float)1;
    }
}
