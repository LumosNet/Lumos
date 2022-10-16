#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#include "gpu.h"

__global__ void fill_kernel(float *data, int len, float x, int offset)
{
    int index = (blockDim.x * blockIdx.x + threadIdx.x)*offset;
    if (index >= len) return;
    data[index] = x;
}

void fill_gpu(float *data, int len, float x, int offset)
{
    fill_kernel<<<(len+BLOCK-1)/BLOCK, BLOCK>>>(data, len, x, offset);
}

void multy_gpu(float *data, int len, float x, int offset);
// void add_gpu(float *data, int len, float x, int offset);

float min_gpu(float *data, int num);
float max_gpu(float *data, int num);
float sum_gpu(float *data, int num);
float mean_gpu(float *data, int num);

void add_gpu(float *data_a, float *data_b, int num, float *space);
void subtract_gpu(float *data_a, float *data_b, int num, float *space);
void multiply_gpu(float *data_a, float *data_b, int num, float *space);
void divide_gpu(float *data_a, float *data_b, int num, float *space);

void saxpy_gpu(float *data_a, float *data_b, int num, float x, float *space);
