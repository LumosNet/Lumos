#include "cpu_call.h"

void call_fill_cpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *len = (int*)params[1];
    float *x = (float*)params[2];
    int *offset = (int*)params[3];
    fill_cpu(data, len[0], x[0], offset[0]);
    ret[0] = (void*)data;
}

void call_multy_cpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *len = (int*)params[1];
    float *x = (float*)params[2];
    int *offset = (int*)params[3];
    multy_cpu(data, len[0], x[0], offset[0]);
    ret[0] = (void*)data;
}

void call_add_cpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *len = (int*)params[1];
    float *x = (float*)params[2];
    int *offset = (int*)params[3];
    add_cpu(data, len[0], x[0], offset[0]);
    ret[0] = (void*)data;
}

void call_min_cpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *num = (int*)params[1];
    float *space = (float*)params[2];
    min_cpu(data, num[0], space);
    ret[0] = (void*)space;
}

void call_max_cpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *num = (int*)params[1];
    float *space = (float*)params[2];
    max_cpu(data, num[0], space);
    ret[0] = (void*)space;
}

void call_sum_cpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *num = (int*)params[1];
    float *space = (float*)params[2];
    sum_cpu(data, num[0], space);
    ret[0] = (void*)space;
}

void call_mean_cpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *num = (int*)params[1];
    float *space = (float*)params[2];
    mean_cpu(data, num[0], space);
    ret[0] = (void*)space;
}

void call_matrix_add_cpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *space = (float*)params[3];
    matrix_add_cpu(data_a, data_b, num[0], space);
    ret[0] = (void*)space;
}

void call_matrix_subtract_cpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *space = (float*)params[3];
    matrix_subtract_cpu(data_a, data_b, num[0], space);
    ret[0] = (void*)space;
}

void call_matrix_multiply_cpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *space = (float*)params[3];
    matrix_multiply_cpu(data_a, data_b, num[0], space);
    ret[0] = (void*)space;
}

void call_matrix_divide_cpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *space = (float*)params[3];
    matrix_divide_cpu(data_a, data_b, num[0], space);
    ret[0] = (void*)space;
}

void call_saxpy_cpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *x = (float*)params[3];
    float *space = (float*)params[4];
    saxpy_cpu(data_a, data_b, num[0], x[0], space);
    ret[0] = (void*)space;
}

void call_sum_channel_cpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    float *ALPHA = (float*)params[4];
    float *space = (float*)params[5];
    sum_channel_cpu(data, h[0], w[0], c[0], ALPHA[0], space);
    ret[0] = (void*)space;
}

void call_one_hot_encoding(void **params, void **ret)
{
    int *n = (int*)params[0];
    int *label = (int*)params[1];
    float *space = (float*)params[2];
    one_hot_encoding(n[0], label[0], space);
    ret[0] = (void*)space;
}
