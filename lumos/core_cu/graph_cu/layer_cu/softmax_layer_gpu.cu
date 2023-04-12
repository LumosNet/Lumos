#include "softmax_layer_gpu.h"

void forward_softmax_layer_gpu(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        softmax_exp_sum_gpu(input, l.inputs, l.workspace, l.workspace+l.inputs);
        softmax_gpu(l.workspace, l.inputs, output, l.workspace[l.inputs]);
    }
}

void backward_softmax_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        float *input = l.input + offset_i;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        softmax_exp_sum_gpu(input, l.inputs, l.workspace, l.workspace+l.inputs);
        softmax_grident_gpu(l.workspace, l.inputs, delta_l, l.workspace[l.inputs]);
        matrix_multiply_gpu(delta_n, delta_l, l.inputs, delta_l);
    }
}