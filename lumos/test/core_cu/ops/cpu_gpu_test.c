#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include "utest.h"
#include "cpu_gpu.h"

void test_matrix_multiply_gpu()
{
    test_run("test_matrix_multiply_gpu");
    float data1[5] = {1, 2, 3, 4, 5};
    float data2[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    float res[5];
    float *data1_g;
    float *data2_g;
    float *res_g;
    cudaMalloc((void**)&data1_g, 5*sizeof(float));
    cudaMalloc((void**)&data2_g, 5*sizeof(float));
    cudaMalloc((void**)&res_g, 5*sizeof(float));
    cudaMemcpy(data1_g, data1, 5*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data2_g, data2, 5*sizeof(float), cudaMemcpyHostToDevice);
    matrix_multiply_gpu(data1_g, data2_g, 5, res_g);
    cudaMemcpy(res, res_g, 5*sizeof(float), cudaMemcpyDeviceToHost);
    if (fabs(res[0]-0.1) > 1e-6){
        test_res(1, "");
    }
    if (fabs(res[1]-0.4) > 1e-6){
        test_res(1, "");
    }
    if (fabs(res[2]-0.9) > 1e-6){
        test_res(1, "");
    }
    if (fabs(res[3]-1.6) > 1e-6){
        test_res(1, "");
    }
    if (fabs(res[4]-2.5) > 1e-6){
        test_res(1, "");
    }
    test_res(0, "");
}

int main()
{
    test_matrix_multiply_gpu();
    return 0;
}
