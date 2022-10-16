#include <stdio.h>
#include <stdlib.h>

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

extern void fill_gpu(float *data, int len, float x, int offset)
{
    dim3 dimGrid(len+BLOCK-1)/BLOCK, 1, 1);
    dim3 dimBlock(BLOCK, 1, 1);
    fill_kernel<<<dimGrid, dimBlock>>>(data, len, x, offset);
}

int main()
{
    float x[10];
    float *data;
    cudaMalloc((void**)&data, 10*sizeof(float));
    fill_gpu(data, 10, 1, 1);
    cudaMemcpy(x, data, 10*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i){
        printf("%f ", x[i]);
    }
    printf("\n");
    add_add_add3(2, 3);
    return 0;
}
