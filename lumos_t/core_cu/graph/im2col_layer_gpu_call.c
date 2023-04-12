#include "im2col_layer_gpu_call.h"

void call_forward_im2col_layer_gpu(void **params, void **ret)
{
    int *flag = (int*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    int *num = (int*)params[4];
    float *input = (float*)params[5];
    float *output = (float*)params[6];
    Layer *l = make_im2col_layer(flag[0]);
    init_im2col_layer(l, w[0], h[0], c[0]);
    float *input_g = NULL;
    float *output_g = NULL;
    cudaMalloc((void**)&input_g, l->inputs*sizeof(float));
    cudaMalloc((void**)&output_g, l->outputs*sizeof(float));
    cudaMemcpy(input_g, input, l->inputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_g, output, l->outputs*sizeof(float), cudaMemcpyHostToDevice);
    l->input = input_g;
    l->output = output_g;
    l->forward(*l, num[0]);
    cudaMemcpy(output, output_g, l->outputs*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input_g);
    cudaFree(output_g);
    ret[0] = output;
}

void call_backward_im2col_layer_gpu(void **params, void **ret)
{
    int *flag = (int*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    int *num = (int*)params[4];
    float *rate = (float*)params[5];
    float *l_delta = (float*)params[6];
    float *n_delta = (float*)params[7];
    Layer *l = make_im2col_layer(flag[0]);
    init_im2col_layer(l, w[0], h[0], c[0]);
    float *l_delta_g = NULL;
    float *n_delta_g = NULL;
    cudaMalloc((void**)&l_delta_g, l->inputs*sizeof(float));
    cudaMalloc((void**)&n_delta_g, l->inputs*sizeof(float));
    cudaMemcpy(l_delta_g, l_delta, l->inputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(n_delta_g, n_delta, l->inputs*sizeof(float), cudaMemcpyHostToDevice);
    l->delta = l_delta_g;
    l->backward(*l, rate[0], num[0], n_delta_g);
    cudaMemcpy(l_delta, l_delta_g, l->inputs*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(l_delta_g);
    cudaFree(n_delta_g);
    ret[0] = l_delta;
}
