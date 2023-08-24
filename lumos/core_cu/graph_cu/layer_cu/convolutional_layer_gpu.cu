#include "convolutional_layer_gpu.h"

void init_convolutional_layer_gpu(Layer *l, int w, int h, int c)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = (l->input_h + 2 * l->pad - l->ksize) / l->stride + 1;
    l->output_w = (l->input_w + 2 * l->pad - l->ksize) / l->stride + 1;
    l->output_c = l->filters;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = l->ksize * l->ksize * l->input_c * l->output_h * l->output_w + l->filters * l->ksize * l->ksize * l->input_c;

    l->forward = forward_convolutional_layer_gpu;
    l->backward = backward_convolutional_layer_gpu;
    l->update = update_convolutional_layer_gpu;

    cudaMalloc((void**)&l->output, l->outputs*l->subdivision*sizeof(float));
    cudaMalloc((void**)&l->delta, l->inputs*l->subdivision*sizeof(float));
    cudaMalloc((void**)&l->kernel_weights, l->filters*l->ksize*l->ksize*l->input_c*sizeof(float));
    cudaMalloc((void**)&l->update_kernel_weights, l->filters*l->ksize*l->ksize*l->input_c*sizeof(float));
    if (l->bias){
        cudaMalloc((void**)&l->bias_weights, l->filters*sizeof(float));
        cudaMalloc((void**)&l->update_bias_weights, l->filters*sizeof(float));
    }

    fprintf(stderr, "Convolutional   Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_convolutional_layer_gpu(Layer l, int num)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        im2col_gpu(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
        gemm_gpu(0, 0, l.filters, l.ksize * l.ksize * l.input_c, l.ksize * l.ksize * l.input_c, l.output_h * l.output_w, 1,
             l.kernel_weights_gpu, l.workspace, output);
        if (l.bias)
        {
            add_bias_gpu(output, l.bias_weights_gpu, l.filters, l.output_h * l.output_w);
        }
        activate_list_gpu(output, l.outputs, l.active);
    }
}

void backward_convolutional_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        gradient_list_gpu(output, l.outputs, l.gradient);
        matrix_multiply_gpu(delta_n, output, l.outputs, delta_n);
        gemm_gpu(1, 0, l.filters, l.ksize * l.ksize * l.input_c,
             l.filters, l.output_h * l.output_w, 1,
             l.kernel_weights_gpu, delta_n, l.workspace);
        col2im_gpu(l.workspace, l.ksize, l.stride, l.pad, l.input_h, l.input_w, l.input_c, delta_l);
    }
    update_convolutional_layer_gpu(l, rate, num, n_delta);
}

void update_convolutional_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_n = n_delta + offset_o;
        im2col_gpu(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
        gemm_gpu(0, 1, l.filters, l.output_h * l.output_w,
             l.ksize * l.ksize * l.input_c, l.output_h * l.output_w, 1,
             delta_n, l.workspace, l.workspace + l.ksize * l.ksize * l.input_c * l.output_h * l.output_w);
        saxpy_gpu(l.update_kernel_weights_gpu, l.workspace + l.ksize * l.ksize * l.input_c * l.output_h * l.output_w, l.filters * l.ksize * l.ksize * l.input_c, rate, l.update_kernel_weights_gpu);
        if (l.bias)
        {
            sum_channel_gpu(delta_n, l.output_h, l.output_w, l.output_c, rate, l.workspace);
            add_bias_gpu(l.update_bias_weights_gpu, l.workspace, l.output_c, l.output_h*l.output_w);
        }
    }
}
