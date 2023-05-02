#include "shortcut_layer_gpu.h"

void forward_shortcut_layer_gpu(Layer l, int num)
{
    Layer *shortcut = l.shortcut;
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        int offset_c = i * shortcut->outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        float *add = shortcut->output + offset_c;
        shortcut_gpu(add, shortcut->output_w, shortcut->output_h, shortcut->output_c, \
                     input, l.input_w, l.input_h, l.input_c, 1, 1, output);
    }
    activate_list_gpu(l.output, num*l.outputs, l.active);
}

void backward_shortcut_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    Layer *shortcut = l.shortcut;
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        int offset_c = i * shortcut->inputs;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        float *out = shortcut->delta + offset_c;
        gradient_list_gpu(output, l.outputs, l.active);
        matrix_multiply_gpu(output, delta_n, l.inputs, delta_l);
        shortcut_gpu(delta_l, l.input_w, l.input_h, l.input_c, \
                     out, shortcut->input_w, shortcut->input_h, shortcut->input_c, \
                     1, 1, out);
    }
}
