#include "cpu_gpu_call.h"

void call_fill_gpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *len = (int*)params[1];
    float *x = (float*)params[2];
    int *offset = (int*)params[3];
    fill_gpu(data, len[0], x[0], offset[0]);
    ret[0] = (void*)data;
}

void call_multy_gpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *len = (int*)params[1];
    float *x = (float*)params[2];
    int *offset = (int*)params[3];
    multy_gpu(data, len[0], x[0], offset[0]);
    ret[0] = (void*)data;
}

void call_add_gpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *len = (int*)params[1];
    float *x = (float*)params[2];
    int *offset = (int*)params[3];
    add_gpu(data, len[0], x[0], offset[0]);
    ret[0] = (void*)data;
}

void call_min_gpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *num = (int*)params[1];
    float *space = (float*)params[2];
    min_gpu(data, num[0], space);
    ret[0] = (void*)space;
}

void call_max_gpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *num = (int*)params[1];
    float *space = (float*)params[2];
    max_gpu(data, num[0], space);
    ret[0] = (void*)space;
}

void call_sum_gpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *num = (int*)params[1];
    float *space = (float*)params[2];
    sum_gpu(data, num[0], space);
    ret[0] = (void*)space;
}

void call_mean_gpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *num = (int*)params[1];
    float *space = (float*)params[2];
    mean_gpu(data, num[0], space);
    ret[0] = (void*)space;
}

void call_matrix_add_gpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *space = (float*)params[3];
    matrix_add_gpu(data_a, data_b, num[0], space);
    ret[0] = (void*)space;
}

void call_matrix_subtract_gpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *space = (float*)params[3];
    matrix_subtract_gpu(data_a, data_b, num[0], space);
    ret[0] = (void*)space;
}

void call_matrix_multiply_gpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *space = (float*)params[3];
    matrix_multiply_gpu(data_a, data_b, num[0], space);
    ret[0] = (void*)space;
}

void call_matrix_divide_gpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *space = (float*)params[3];
    matrix_divide_gpu(data_a, data_b, num[0], space);
    ret[0] = (void*)space;
}

void call_saxpy_gpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *x = (float*)params[3];
    float *space = (float*)params[4];
    saxpy_gpu(data_a, data_b, num[0], x[0], space);
    ret[0] = (void*)space;
}

void call_sum_channel_gpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    float *ALPHA = (float*)params[4];
    float *space = (float*)params[5];
    sum_channel_gpu(data, h[0], w[0], c[0], ALPHA[0], space);
    ret[0] = (void*)space;
}
