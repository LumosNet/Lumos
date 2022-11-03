#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include "utest.h"
#include "im2col_layer.h"
#include "im2col_layer_gpu.h"

void test_forward_im2col_layer_gpu()
{
    test_run("test_forward_im2col_layer_gpu");
    Layer *l = make_im2col_layer(1);
    init_im2col_layer(l, 1, 5, 1);
    float input[5] = {1, 2, 3, 4, 5};
    float output[5];
    float *input_g;
    float *output_g;
    cudaMalloc((void**)&input_g, 5*sizeof(float));
    cudaMalloc((void**)&output_g, 5*sizeof(float));
    cudaMemcpy(input_g, input, 5*sizeof(float), cudaMemcpyHostToDevice);
    l->input = input_g;
    l->output = output_g;
    forward_im2col_layer_gpu(*l, 1);
    cudaMemcpy(output, output_g, 5*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; ++i){
        if (fabs(output[i]-input[i]) > 1e-6){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

void test_backward_im2col_layer_gpu()
{
    test_run("test_backward_im2col_layer_gpu");
    Layer *l = make_im2col_layer(1);
    init_im2col_layer(l, 1, 5, 1);
    float delta[5];
    float n_delta[5] = {1, 2, 3, 4, 5};
    float *delta_g;
    float *n_delta_g;
    cudaMalloc((void**)&delta_g, 5*sizeof(float));
    cudaMalloc((void**)&n_delta_g, 5*sizeof(float));
    cudaMemcpy(n_delta_g, n_delta, 5*sizeof(float), cudaMemcpyHostToDevice);
    l->delta = delta_g;
    backward_im2col_layer_gpu(*l, 1, 1, n_delta_g);
    cudaMemcpy(delta, delta_g, 5*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; ++i){
        if (fabs(delta[i]-n_delta[i]) > 1e-6){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

int main()
{
    test_forward_im2col_layer_gpu();
    test_backward_im2col_layer_gpu();
    return 0;
}
