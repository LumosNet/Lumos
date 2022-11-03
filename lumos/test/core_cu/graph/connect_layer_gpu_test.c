#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include "connect_layer_gpu.h"

void test_forward_connect_layer_gpu()
{
    test_run("test_forward_connect_layer");
    Layer *l;
    l = make_connect_layer(4, 1, "relu", "guass");
    init_connect_layer(l, 1, 2, 1);
    float *input_cpu = malloc(2*sizeof(float));
    input_cpu[0] = 1;   // 1
    input_cpu[1] = 2;   // 2
    float *output_cpu = calloc(4, sizeof(float));
    float *kernel_weights_cpu = calloc(8, sizeof(float));
    float *update_kernel_weights_cpu = calloc(8, sizeof(float));
    float *bias_weights_cpu = calloc(4, sizeof(float));
    float *update_bias_weights_cpu = calloc(4, sizeof(float));
    float *workspace_cpu = calloc(l->workspace_size, sizeof(float));
    kernel_weights[0] = 0.1;    // 0.1  0.2  1
    kernel_weights[1] = 0.2;    // 0.3  0.4  2
    kernel_weights[2] = 0.3;    // 0.5  0.6
    kernel_weights[3] = 0.4;    // 0.7  0.8
    kernel_weights[4] = 0.5;
    kernel_weights[5] = 0.6;
    kernel_weights[6] = 0.7;
    kernel_weights[7] = 0.8;
    memcpy(update_kernel_weights, kernel_weights, 8*sizeof(float));
    bias_weights[0] = 0.01;
    bias_weights[1] = 0.01;
    bias_weights[2] = 0.01;
    bias_weights[3] = 0.01;
    memcpy(update_bias_weights, bias_weights, 4*sizeof(float));
    float *input;
    float *output;
    float *kernel_weights;
    float *update_kernel_weights;
    float *bias_weights;
    float *update_bias_weights;
    float *workspace;
    cudaMalloc((void**)&input, 2*sizeof(float));
    cudaMalloc((void**)&output, 4*sizeof(float));
    cudaMalloc((void**)&kernel_weights, 8*sizeof(float));
    cudaMalloc((void**)&update_kernel_weights, 8*sizeof(float));
    cudaMalloc((void**)&bias_weights, 4*sizeof(float));
    cudaMalloc((void**)&update_bias_weights, 4*sizeof(float));
    cudaMalloc((void**)&workspace, sizeof(float));
    cudaMemcpy(input, input_cpu, 2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output, output_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_weights, kernel_weights_cpu, 8*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(update_kernel_weights, update_kernel_weights_cpu, 8*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_weights, bias_weights_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(update_bias_weights, update_bias_weights_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(workspace, workspace_cpu, sizeof(float), cudaMemcpyHostToDevice);
    l->input = input;
    l->output = output;
    l->workspace = workspace;
    l->kernel_weights = kernel_weights;
    l->update_kernel_weights = update_kernel_weights;
    l->bias_weights = bias_weights;
    l->update_bias_weights = update_bias_weights;
    /*
        0.5  0.51
        1.1  1.11
        1.7  1.71
        2.3  2.31
    */
    forward_connect_layer(*l, 1);
    cudaMemcpy(output_cpu, output, 4*sizeof(float), cudaMemcpyDeviceToHost);
    if (fabs(output_cpu[0]-0.51) > 1e-6){
        test_res(1, "");
        return;
    }
    if (fabs(output_cpu[1]-1.11) > 1e-6){
        test_res(1, "");
        return;
    }
    if (fabs(output_cpu[2]-1.71) > 1e-6){
        test_res(1, "");
        return;
    }
    if (fabs(output_cpu[3]-2.31) > 1e-6){
        test_res(1, "");
        return;
    }
    test_res(0, "");
}