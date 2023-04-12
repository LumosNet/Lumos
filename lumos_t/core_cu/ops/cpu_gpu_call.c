#include "cpu_gpu_call.h"

void call_fill_gpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *len = (int*)params[1];
    float *x = (float*)params[2];
    int *offset = (int*)params[3];
    float *data_gpu = NULL;
    cudaMalloc((void**)&data_gpu, len[0]*sizeof(float));
    cudaMemcpy(data_gpu, data, len[0]*sizeof(float), cudaMemcpyHostToDevice);
    fill_gpu(data_gpu, len[0], x[0], offset[0]);
    cudaMemcpy(data, data_gpu, len[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(data_gpu);
    ret[0] = (void*)data;
}

void call_multy_gpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *len = (int*)params[1];
    float *x = (float*)params[2];
    int *offset = (int*)params[3];
    float *data_gpu = NULL;
    cudaMalloc((void**)&data_gpu, len[0]*sizeof(float));
    cudaMemcpy(data_gpu, data, len[0]*sizeof(float), cudaMemcpyHostToDevice);
    multy_gpu(data_gpu, len[0], x[0], offset[0]);
    cudaMemcpy(data, data_gpu, len[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(data_gpu);
    ret[0] = (void*)data;
}

void call_add_gpu(void **params, void **ret)
{
    float *data = (float*)params[0];
    int *len = (int*)params[1];
    float *x = (float*)params[2];
    int *offset = (int*)params[3];
    float *data_gpu = NULL;
    cudaMalloc((void**)&data_gpu, len[0]*sizeof(float));
    cudaMemcpy(data_gpu, data, len[0]*sizeof(float), cudaMemcpyHostToDevice);
    add_gpu(data_gpu, len[0], x[0], offset[0]);
    cudaMemcpy(data, data_gpu, len[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(data_gpu);
    ret[0] = (void*)data;
}

void call_matrix_add_gpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *space = (float*)params[3];
    float *data_a_g = NULL;
    float *data_b_g = NULL;
    float *space_g = NULL;
    cudaMalloc((void**)&data_a_g, num[0]*sizeof(float));
    cudaMalloc((void**)&data_b_g, num[0]*sizeof(float));
    cudaMalloc((void**)&space_g, num[0]*sizeof(float));
    cudaMemcpy(data_a_g, data_a, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_b_g, data_b, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(space_g, space, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    matrix_add_gpu(data_a_g, data_b_g, num[0], space_g);
    cudaMemcpy(space, space_g, num[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(data_a_g);
    cudaFree(data_b_g);
    cudaFree(space_g);
    ret[0] = (void*)space;
}

void call_matrix_subtract_gpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *space = (float*)params[3];
    float *data_a_g = NULL;
    float *data_b_g = NULL;
    float *space_g = NULL;
    cudaMalloc((void**)&data_a_g, num[0]*sizeof(float));
    cudaMalloc((void**)&data_b_g, num[0]*sizeof(float));
    cudaMalloc((void**)&space_g, num[0]*sizeof(float));
    cudaMemcpy(data_a_g, data_a, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_b_g, data_b, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(space_g, space, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    matrix_subtract_gpu(data_a_g, data_b_g, num[0], space_g);
    cudaMemcpy(space, space_g, num[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(data_a_g);
    cudaFree(data_b_g);
    cudaFree(space_g);
    ret[0] = (void*)space;
}

void call_matrix_multiply_gpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *space = (float*)params[3];
    float *data_a_g = NULL;
    float *data_b_g = NULL;
    float *space_g = NULL;
    cudaMalloc((void**)&data_a_g, num[0]*sizeof(float));
    cudaMalloc((void**)&data_b_g, num[0]*sizeof(float));
    cudaMalloc((void**)&space_g, num[0]*sizeof(float));
    cudaMemcpy(data_a_g, data_a, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_b_g, data_b, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(space_g, space, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    matrix_multiply_gpu(data_a_g, data_b_g, num[0], space_g);
    cudaMemcpy(space, space_g, num[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(data_a_g);
    cudaFree(data_b_g);
    cudaFree(space_g);
    ret[0] = (void*)space;
}

void call_matrix_divide_gpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *space = (float*)params[3];
    float *data_a_g = NULL;
    float *data_b_g = NULL;
    float *space_g = NULL;
    cudaMalloc((void**)&data_a_g, num[0]*sizeof(float));
    cudaMalloc((void**)&data_b_g, num[0]*sizeof(float));
    cudaMalloc((void**)&space_g, num[0]*sizeof(float));
    cudaMemcpy(data_a_g, data_a, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_b_g, data_b, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(space_g, space, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    matrix_divide_gpu(data_a_g, data_b_g, num[0], space_g);
    cudaMemcpy(space, space_g, num[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(data_a_g);
    cudaFree(data_b_g);
    cudaFree(space_g);
    ret[0] = (void*)space;
}

void call_saxpy_gpu(void **params, void **ret)
{
    float *data_a = (float*)params[0];
    float *data_b = (float*)params[1];
    int *num = (int*)params[2];
    float *x = (float*)params[3];
    float *space = (float*)params[4];
    float *data_a_g = NULL;
    float *data_b_g = NULL;
    float *space_g = NULL;
    cudaMalloc((void**)&data_a_g, num[0]*sizeof(float));
    cudaMalloc((void**)&data_b_g, num[0]*sizeof(float));
    cudaMalloc((void**)&space_g, num[0]*sizeof(float));
    cudaMemcpy(data_a_g, data_a, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_b_g, data_b, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(space_g, space, num[0]*sizeof(float), cudaMemcpyHostToDevice);
    saxpy_gpu(data_a_g, data_b_g, num[0], x[0], space_g);
    cudaMemcpy(space, space_g, num[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(data_a_g);
    cudaFree(data_b_g);
    cudaFree(space_g);
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
    float *data_g = NULL;
    float *space_g = NULL;
    cudaMalloc((void**)&data_g, h[0]*w[0]*c[0]*sizeof(float));
    cudaMalloc((void**)&space_g, c[0]*sizeof(float));
    cudaMemcpy(data_g, data, h[0]*w[0]*c[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(space_g, space, c[0]*sizeof(float), cudaMemcpyHostToDevice);
    sum_channel_gpu(data_g, h[0], w[0], c[0], ALPHA[0], space_g);
    cudaMemcpy(space, space_g, c[0]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(data_g);
    cudaFree(space_g);
    ret[0] = (void*)space;
}
