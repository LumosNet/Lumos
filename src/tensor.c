#include "tensor.h"

float get_pixel(float *data, int dim, int *size, int *index)
{
    int num = multing_int_list(size, 0, dim);
    int ts2ls = index_ts2ls(index, dim, size);
    if (ts2ls >= 0 && ts2ls < num) return data[ts2ls];
    return 0;
}

void change_pixel(float *data, int dim, int *size, int *index, float x)
{
    int ts2ls = index_ts2ls(index, dim, size);
    if (ts2ls) data[ts2ls] = x;
}

float min(float *data, int num)
{
    float min = data[0];
    for (int i = 1; i < num; ++i){
        if (data[i] < min) min = data[i];
    }
    return min;
}

float max(float *data, int num)
{
    float max = data[0];
    for (int i = 1; i < num; ++i){
        if (data[i] > max) max = data[i];
    }
    return max;
}

float mean(float *data, int num)
{
    float sum = sum_float_list(data, 0, num);
    return sum / (float)num;
}

void add_x(float *data, int num, float x)
{
    for (int i = 0; i < num; ++i){
        data[i] += x;
    }
}

void mult_x(float *data, int num, float x)
{
    for (int i = 0; i < num; ++i){
        data[i] *= x;
    }
}

void add(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i){
        space[i] = data_a[i] + data_b[i];
    }
}

void subtract(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i){
        space[i] = data_a[i] - data_b[i];
    }
}

void multiply(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i){
        space[i] = data_a[i] * data_b[i];
    }
}

void divide(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i){
        space[i] = data_a[i] / data_b[i];
    }
}

void saxpy(float *data_a, float *data_b, int num, float x, float *space)
{
    for (int i = 0; i < num; ++i){
        space[i] = data_a[i] + data_b[i]*x;
    }
}