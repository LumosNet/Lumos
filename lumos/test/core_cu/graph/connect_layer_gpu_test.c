#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include "utest.h"
#include "connect_layer.h"
#include "connect_layer_gpu.h"

void test_forward_connect_layer_gpu()
{
    test_run("test_forward_connect_layer_gpu");
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
    kernel_weights_cpu[0] = 0.1;    // 0.1  0.2  1
    kernel_weights_cpu[1] = 0.2;    // 0.3  0.4  2
    kernel_weights_cpu[2] = 0.3;    // 0.5  0.6
    kernel_weights_cpu[3] = 0.4;    // 0.7  0.8
    kernel_weights_cpu[4] = 0.5;
    kernel_weights_cpu[5] = 0.6;
    kernel_weights_cpu[6] = 0.7;
    kernel_weights_cpu[7] = 0.8;
    memcpy(update_kernel_weights_cpu, kernel_weights_cpu, 8*sizeof(float));
    bias_weights_cpu[0] = 0.01;
    bias_weights_cpu[1] = 0.01;
    bias_weights_cpu[2] = 0.01;
    bias_weights_cpu[3] = 0.01;
    memcpy(update_bias_weights_cpu, bias_weights_cpu, 4*sizeof(float));
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
    cudaMalloc((void**)&workspace, l->workspace_size*sizeof(float));
    cudaMemcpy(input, input_cpu, 2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output, output_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_weights, kernel_weights_cpu, 8*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(update_kernel_weights, update_kernel_weights_cpu, 8*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_weights, bias_weights_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(update_bias_weights, update_bias_weights_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(workspace, workspace_cpu, l->workspace_size*sizeof(float), cudaMemcpyHostToDevice);
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
    forward_connect_layer_gpu(*l, 1);
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

void test_backward_connect_layer_gpu()
{
    test_run("test_backward_connect_layer_gpu");
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
    float *delta_l_cpu = calloc(2, sizeof(float));
    float *delta_n_cpu = calloc(4, sizeof(float));
    kernel_weights_cpu[0] = 0.1;    // 0.1  0.2  1
    kernel_weights_cpu[1] = 0.2;    // 0.3  0.4  2
    kernel_weights_cpu[2] = 0.3;    // 0.5  0.6
    kernel_weights_cpu[3] = 0.4;    // 0.7  0.8
    kernel_weights_cpu[4] = 0.5;
    kernel_weights_cpu[5] = 0.6;
    kernel_weights_cpu[6] = 0.7;
    kernel_weights_cpu[7] = 0.8;
    memcpy(update_kernel_weights_cpu, kernel_weights_cpu, 8*sizeof(float));
    bias_weights_cpu[0] = 0.01;
    bias_weights_cpu[1] = 0.01;
    bias_weights_cpu[2] = 0.01;
    bias_weights_cpu[3] = 0.01;
    memcpy(update_bias_weights_cpu, bias_weights_cpu, 4*sizeof(float));
    delta_n_cpu[0] = 0.1;           // 0.1
    delta_n_cpu[1] = 0.2;           // 0.2
    delta_n_cpu[2] = 0.3;           // 0.3
    delta_n_cpu[3] = 0.4;           // 0.4
    float *input;
    float *output;
    float *workspace;
    float *kernel_weights;
    float *update_kernel_weights;
    float *bias_weights;
    float *update_bias_weights;
    float *delta_l;
    float *delta_n;
    cudaMalloc((void**)&input, 2*sizeof(float));
    cudaMalloc((void**)&output, 4*sizeof(float));
    cudaMalloc((void**)&workspace, l->workspace_size*sizeof(float));
    cudaMalloc((void**)&kernel_weights, 8*sizeof(float));
    cudaMalloc((void**)&update_kernel_weights, 8*sizeof(float));
    cudaMalloc((void**)&bias_weights, 4*sizeof(float));
    cudaMalloc((void**)&update_bias_weights, 4*sizeof(float));
    cudaMalloc((void**)&delta_l, 2*sizeof(float));
    cudaMalloc((void**)&delta_n, 4*sizeof(float));
    cudaMemcpy(input, input_cpu, 2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output, output_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(workspace, workspace_cpu, l->workspace_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_weights, kernel_weights_cpu, 8*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(update_kernel_weights, update_kernel_weights_cpu, 8*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_weights, bias_weights_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(update_bias_weights, update_bias_weights_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(delta_l, delta_l_cpu, 2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(delta_n, delta_n_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);
    l->input = input;
    l->output = output;
    l->workspace = workspace;
    l->kernel_weights = kernel_weights;
    l->update_kernel_weights = update_kernel_weights;
    l->bias_weights = bias_weights;
    l->update_bias_weights = update_bias_weights;
    l->delta = delta_l;
    /*
        0.5  0.51
        1.1  1.11
        1.7  1.71
        2.3  2.31
    */
    forward_connect_layer_gpu(*l, 1);
    /*
        0.1 0.2 0.3 0.4     0.1
        0.5 0.6 0.7 0.8     0.2
                            0.3
                            0.4

        0.3   0.7
    */
    backward_connect_layer_gpu(*l, 1, 1, delta_n);
    cudaMemcpy(delta_l_cpu, delta_l, 2*sizeof(float), cudaMemcpyDeviceToHost);
    if (fabs(delta_l_cpu[0]-0.3) > 1e-6){
        printf("delta[0]: %f\n", delta_l_cpu[0]);
        test_res(1, "");
    }
    if (fabs(delta_l_cpu[1]-0.7) > 1e-6){
        printf("delta[1]: %f\n", delta_l_cpu[1]);
        test_res(1, "");
    }
    test_res(0, "");
}

void test_update_connect_layer_gpu()
{
    test_run("test_update_connect_layer_gpu");
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
    float *delta_l_cpu = calloc(2, sizeof(float));
    float *delta_n_cpu = calloc(4, sizeof(float));
    kernel_weights_cpu[0] = 0.1;    // 0.1  0.2  1
    kernel_weights_cpu[1] = 0.2;    // 0.3  0.4  2
    kernel_weights_cpu[2] = 0.3;    // 0.5  0.6
    kernel_weights_cpu[3] = 0.4;    // 0.7  0.8
    kernel_weights_cpu[4] = 0.5;
    kernel_weights_cpu[5] = 0.6;
    kernel_weights_cpu[6] = 0.7;
    kernel_weights_cpu[7] = 0.8;
    memcpy(update_kernel_weights_cpu, kernel_weights_cpu, 8*sizeof(float));
    bias_weights_cpu[0] = 0.01;
    bias_weights_cpu[1] = 0.01;
    bias_weights_cpu[2] = 0.01;
    bias_weights_cpu[3] = 0.01;
    memcpy(update_bias_weights_cpu, bias_weights_cpu, 4*sizeof(float));
    delta_n_cpu[0] = 0.1;           // 0.1
    delta_n_cpu[1] = 0.2;           // 0.2
    delta_n_cpu[2] = 0.3;           // 0.3
    delta_n_cpu[3] = 0.4;           // 0.4
    float *input;
    float *output;
    float *workspace;
    float *kernel_weights;
    float *update_kernel_weights;
    float *bias_weights;
    float *update_bias_weights;
    float *delta_l;
    float *delta_n;
    cudaMalloc((void**)&input, 2*sizeof(float));
    cudaMalloc((void**)&output, 4*sizeof(float));
    cudaMalloc((void**)&workspace, l->workspace_size*sizeof(float));
    cudaMalloc((void**)&kernel_weights, 8*sizeof(float));
    cudaMalloc((void**)&update_kernel_weights, 8*sizeof(float));
    cudaMalloc((void**)&bias_weights, 4*sizeof(float));
    cudaMalloc((void**)&update_bias_weights, 4*sizeof(float));
    cudaMalloc((void**)&delta_l, 2*sizeof(float));
    cudaMalloc((void**)&delta_n, 4*sizeof(float));
    cudaMemcpy(input, input_cpu, 2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output, output_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(workspace, workspace_cpu, l->workspace_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_weights, kernel_weights_cpu, 8*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(update_kernel_weights, update_kernel_weights_cpu, 8*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_weights, bias_weights_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(update_bias_weights, update_bias_weights_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(delta_l, delta_l_cpu, 2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(delta_n, delta_n_cpu, 4*sizeof(float), cudaMemcpyHostToDevice);
    l->input = input;
    l->output = output;
    l->workspace = workspace;
    l->kernel_weights = kernel_weights;
    l->update_kernel_weights = update_kernel_weights;
    l->bias_weights = bias_weights;
    l->update_bias_weights = update_bias_weights;
    l->delta = delta_l;
    /*
        0.5  0.51
        1.1  1.11
        1.7  1.71
        2.3  2.31
    */
    forward_connect_layer_gpu(*l, 1);
    /*
        0.1     1 2
        0.2
        0.3
        0.4

        0.1   0.2
        0.2   0.4
        0.3   0.6
        0.4   0.8

        n_k_w:
        0.2
        0.4
        0.5
        0.8
        0.8
        1.2
        1.1
        1.6

        n_b_w:
        0.11
        0.21
        0.31
        0.41
    */
    backward_connect_layer_gpu(*l, 1, 1, delta_n);
    cudaMemcpy(update_kernel_weights_cpu, update_kernel_weights, 8*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(update_bias_weights_cpu, update_bias_weights, 4*sizeof(float), cudaMemcpyDeviceToHost);
    if (fabs(update_kernel_weights_cpu[0]-0.2) > 1e-6){
        printf("delta[0]: %f\n", update_kernel_weights_cpu[0]);
        test_res(1, "");
    }
    if (fabs(update_kernel_weights_cpu[1]-0.4) > 1e-6){
        printf("delta[1]: %f\n", update_kernel_weights_cpu[1]);
        test_res(1, "");
    }
    if (fabs(update_kernel_weights_cpu[2]-0.5) > 1e-6){
        printf("delta[1]: %f\n", update_kernel_weights_cpu[1]);
        test_res(1, "");
    }
    if (fabs(update_kernel_weights_cpu[3]-0.8) > 1e-6){
        printf("delta[1]: %f\n", update_kernel_weights_cpu[1]);
        test_res(1, "");
    }
    if (fabs(update_kernel_weights_cpu[4]-0.8) > 1e-6){
        printf("delta[1]: %f\n", update_kernel_weights_cpu[1]);
        test_res(1, "");
    }
    if (fabs(update_kernel_weights_cpu[5]-1.2) > 1e-6){
        printf("delta[1]: %f\n", update_kernel_weights_cpu[1]);
        test_res(1, "");
    }
    if (fabs(update_kernel_weights_cpu[6]-1.1) > 1e-6){
        printf("delta[1]: %f\n", update_kernel_weights_cpu[1]);
        test_res(1, "");
    }
    if (fabs(update_kernel_weights_cpu[7]-1.6) > 1e-6){
        printf("delta[1]: %f\n", update_kernel_weights_cpu[1]);
        test_res(1, "");
    }
    if (fabs(update_bias_weights_cpu[0]-0.11) > 1e-6){
        printf("delta[1]: %f\n", update_bias_weights_cpu[1]);
        test_res(1, "");
    }
    if (fabs(update_bias_weights_cpu[1]-0.21) > 1e-6){
        printf("delta[1]: %f\n", update_bias_weights_cpu[1]);
        test_res(1, "");
    }
    if (fabs(update_bias_weights_cpu[2]-0.31) > 1e-6){
        printf("delta[1]: %f\n", update_bias_weights_cpu[1]);
        test_res(1, "");
    }
    if (fabs(update_bias_weights_cpu[3]-0.41) > 1e-6){
        printf("delta[1]: %f\n", update_bias_weights_cpu[1]);
        test_res(1, "");
    }
    test_res(0, "");
}

int main()
{
    test_forward_connect_layer_gpu();
    test_backward_connect_layer_gpu();
    test_update_connect_layer_gpu();
    return 0;
}
