#include "im2col_layer_gpu.h"

void init_im2col_layer_gpu(Layer *l, int w, int h, int c)
{
    l->input_h = h,
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = 1;
    l->output_w = 1;
    l->output_c = l->inputs;
    l->outputs = l->inputs;
    l->workspace_size = 0;

    l->forward = forward_im2col_layer_gpu;
    l->backward = backward_im2col_layer_gpu;

    cudaMalloc((void**)&l->output, l->outputs*l->subdivision*sizeof(float));
    cudaMalloc((void**)&l->delta, l->inputs*l->subdivision*sizeof(float));

    fprintf(stderr, "Im2col          Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_im2col_layer_gpu(Layer l, int num)
{
    cudaMemcpy(l.output, l.input, num*l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
}

void backward_im2col_layer_gpu(Layer l, float rate, int num)
{
    cudaMemcpy(l.delta, l.n_delta, num*l.inputs*sizeof(float), cudaMemcpyDeviceToDevice);
}
