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

void add(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i)
    {
        space[i] = data_a[i] + data_b[i];
    }
}

void subtract(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i)
    {
        space[i] = data_a[i] - data_b[i];
    }
}

void multiply(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i)
    {
        space[i] = data_a[i] * data_b[i];
    }
}

void divide(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i)
    {
        space[i] = data_a[i] / data_b[i];
    }
}

void saxpy(float *data_a, float *data_b, int num, float x, float *space)
{
    for (int i = 0; i < num; ++i)
    {
        space[i] = data_a[i] + data_b[i] * x;
    }
}

void random(int range_l, int range_r, float scale, int num, float *space)
{
    for (int i = 0; i < num; ++i)
    {
        space[i] = (rand() % (range_r + 1) + range_l) * scale;
    }
}
