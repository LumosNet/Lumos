#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include "bias_gpu.h"
#include "cpu_gpu.h"
#include "tsession.h"

int main(void)
{
    // run_benchmarks("./lumos_t/benchmark/core_cu/ops/bias/add_bias_gpu.json");
    float *origin = (float*)malloc(sizeof(float));
    float *bias = (float*)malloc(sizeof(float));
    origin[0] = 1;
    bias[0] = 2;
    float *origin_gpu = NULL;
    float *bias_gpu = NULL;
    cudaMalloc((void**)&origin_gpu, sizeof(float));
    cudaMalloc((void**)&bias_gpu, sizeof(float));
    cudaMemcpy(origin, origin_gpu, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias, bias_gpu, sizeof(float), cudaMemcpyHostToDevice);
    // add_bias_gpu(origin_gpu, bias_gpu, 1, 1);
    add_gpu(origin_gpu, 1, 0.1, 1);
    cudaMemcpy(origin_gpu, origin, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", origin[0]);
    return 0;
}
