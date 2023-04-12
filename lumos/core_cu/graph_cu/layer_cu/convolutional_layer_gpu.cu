#include "convolutional_layer_gpu.h"

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
    l.update(l, rate, num, n_delta);
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
