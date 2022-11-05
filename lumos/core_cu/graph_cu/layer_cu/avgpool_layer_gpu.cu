#include "avgpool_layer_gpu.h"

void forward_avgpool_layer_gpu(Layer l, int num)
{
    fill_gpu(l.output, l.outputs * num, 0, 1);
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        avgpool_gpu(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, output);
    }
}

void backward_avgpool_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        avgpool_gradient_gpu(delta_l, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, delta_n);
    }
}
