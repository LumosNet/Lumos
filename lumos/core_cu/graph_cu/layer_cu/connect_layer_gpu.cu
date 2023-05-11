#include "connect_layer_gpu.h"

void forward_connect_layer_gpu(Layer l, int num)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        gemm_gpu(0, 0, l.outputs, l.inputs, l.inputs, 1,
             1, l.kernel_weights_gpu, input, output);
        if (l.batchnorm){
            forward_normalization_layer_gpu(l, num);
        }
        if (l.bias)
        {
            add_bias_gpu(output, l.bias_weights_gpu, l.ksize, 1);
        }
        activate_list_gpu(output, l.outputs, l.active);
    }
}

void backward_connect_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        gradient_list_gpu(output, l.outputs, l.gradient);
        if (l.batchnorm){
            backward_normalization_layer_gpu(l, rate, num, n_delta);
        }
        matrix_multiply_gpu(delta_n, output, l.outputs, delta_n);
        gemm_gpu(1, 0, l.output_c, l.input_c, l.output_c, l.input_w, 1,
             l.kernel_weights_gpu, delta_n, delta_l);
    }
    l.update(l, rate, num, n_delta);
}

void update_connect_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_n = n_delta + offset_o;
        gemm_gpu(0, 1, l.output_c, l.output_w,
             l.input_c, l.input_w, 1,
             delta_n, input, l.workspace);
        saxpy_gpu(l.update_kernel_weights_gpu, l.workspace, l.output_c * l.input_c, rate, l.update_kernel_weights_gpu);
        if (l.bias)
        {
            saxpy_gpu(l.update_bias_weights_gpu, delta_n, l.outputs, rate, l.update_bias_weights_gpu);
        }
    }
}
