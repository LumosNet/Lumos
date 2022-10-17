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

void add(float *data, int len, float x, int offset)
{
    for (int i = 0; i < len; i += offset)
    {
        data[i] += x;
    }
}

float min_cpu(float *data, int num)
{
    float min = data[0];
    for (int i = 1; i < num; ++i)
    {
        if (data[i] < min)
            min = data[i];
    }
    return min;
}

float max_cpu(float *data, int num)
{
    float max = data[0];
    for (int i = 1; i < num; ++i)
    {
        if (data[i] > max)
            max = data[i];
    }
    return max;
}

float sum_cpu(float *data, int num)
{
    float res = 0;
    for (int i = 0; i < num; ++i)
    {
        res += data[i];
    }
    return res;
}

float mean_cpu(float *data, int num)
{
    float sum = sum_cpu(data, num);
    return sum / (float)num;
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
