#include "avgpool_layer_gpu_call.h"

void call_forward_avgpool_layer_gpu(void **params, void **ret)
{
    int *ksize = (int*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    int *num = (int*)params[4];
    float *input = (float*)params[5];
    float *output = (float*)params[6];
    float *input_g = NULL;
    float *output_g = NULL;
    Layer *l = make_avgpool_layer(ksize[0]);
    init_avgpool_layer(l, w[0], h[0], c[0]);
    cudaMalloc((void**)&input_g, num[0]*l->inputs*sizeof(float));
    cudaMalloc((void**)&output_g, num[0]*l->outputs*sizeof(float));
    cudaMemcpy(input_g, input, num[0]*l->inputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_g, output, num[0]*l->outputs*sizeof(float), cudaMemcpyHostToDevice);
    l->input = input_g;
    l->output = output_g;
    l->forward(*l, num[0]);
    cudaMemcpy(output, output_g, num[0]*l->outputs*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input_g);
    cudaFree(output_g);
    ret[0] = output;
}

void call_backward_avgpool_layer_gpu(void **params, void **ret)
{
    int *ksize = (int*)params[0];
    int *h = (int*)params[1];
    int *w = (int*)params[2];
    int *c = (int*)params[3];
    float *rate = (float*)params[4];
    int *num = (int*)params[5];
    float *n_delta = (float*)params[6];
    float *l_delta = (float*)params[7];
    float *n_delta_g = NULL;
    float *l_delta_g = NULL;
    Layer *l = make_avgpool_layer(ksize[0]);
    init_avgpool_layer(l, w[0], h[0], c[0]);
    cudaMalloc((void**)&n_delta_g, num[0]*l->outputs*sizeof(float));
    cudaMalloc((void**)&l_delta_g, num[0]*l->inputs*sizeof(float));
    cudaMemcpy(n_delta_g, n_delta, num[0]*l->outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l_delta_g, l_delta, num[0]*l->inputs*sizeof(float), cudaMemcpyHostToDevice);
    l->delta = l_delta_g;
    l->backward(*l, rate[0], num[0], n_delta_g);
    cudaMemcpy(l_delta, l_delta_g, num[0]*l->inputs*sizeof(float), cudaMemcpyDeviceToHost);
    ret[0] = l_delta;
}
