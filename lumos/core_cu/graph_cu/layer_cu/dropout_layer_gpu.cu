#include "dropout_layer_gpu.h"

void init_dropout_layer_gpu(Layer *l, int w, int h, int c)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = h;
    l->output_w = w;
    l->output_c = c;
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->workspace_size = l->inputs;

    l->forward = forward_dropout_layer_gpu;
    l->backward = backward_dropout_layer_gpu;

    cudaMalloc((void**)&l->output, l->outputs*l->subdivision*sizeof(float));
    cudaMalloc((void**)&l->delta, l->inputs*l->subdivision*sizeof(float));

    fprintf(stderr, "Dropout         Layer\n");
}

void forward_dropout_layer_gpu(Layer l, int num)
{
    if (!l.train){
        cudaMemcpy(l.output, l.input, num*l.inputs*sizeof(float), cudaMemcpyDeviceToDevice);
        return;
    }
    dropout_gpu(l, num);
}

void backward_dropout_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    if (!l.train){
        cudaMemcpy(l.delta, n_delta, num*l.inputs*sizeof(float), cudaMemcpyDeviceToDevice);
        return;
    }
    dropout_gradient_gpu(l, num, n_delta);
}

__device__ float rand_uniform_gpu(float a, float b, int seed)
{
	float t;
	seed = 2045.0 * seed + 1;
	seed = seed - (seed / 1048576) * 1048576;
	t = seed / 1048576.0;
	t = a + (b - a) * t;
	return t;
}

__global__ void dropout_kernel(Layer l, int num, float scale)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num*l.inputs) return;
    float r = rand_uniform_gpu(0, 1, index);
    l.dropout_rand[index] = r;
    if (r < l.probability) l.output[index] = 0;
    else l.output[index] = l.input[index] * scale;
}

__global__ void dropout_gradient_kernel(Layer l, int num, float *n_delta, float scale)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num*l.inputs) return;
    float r = l.dropout_rand[index];
    if (r < l.probability) l.delta[index] = 0;
    else l.delta[index] = n_delta[index] * scale;
}

void dropout_gpu(Layer l, int num)
{
    int size = num * l.inputs;
    float scale = 1. / (1.-l.probability);
    dropout_kernel<<<(size+BLOCK-1)/BLOCK, BLOCK>>>(l, num, scale);
}

void dropout_gradient_gpu(Layer l, int num, float *n_delta)
{
    int size = num * l.inputs;
    float scale = 1. / (1.-l.probability);
    dropout_gradient_kernel<<<(size+BLOCK-1)/BLOCK, BLOCK>>>(l, num, n_delta, scale);
}
