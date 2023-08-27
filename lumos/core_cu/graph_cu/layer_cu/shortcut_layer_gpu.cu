#include "shortcut_layer_gpu.h"

void init_shortcut_layer_gpu(Layer *l, int w, int h, int c, Layer *shortcut)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = h;
    l->output_w = w;
    l->output_c = c;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->shortcut = shortcut;
    l->workspace_size = 0;

    l->forward = forward_shortcut_layer_gpu;
    l->backward = backward_shortcut_layer_gpu;

    cudaMalloc((void**)&l->output, l->outputs*l->subdivision*sizeof(float));
    cudaMalloc((void**)&l->delta, l->inputs*l->subdivision*sizeof(float));

    fprintf(stderr, "Shortcut        Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

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

void backward_shortcut_layer_gpu(Layer l, float rate, int num)
{
    Layer *shortcut = l.shortcut;
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        int offset_c = i * shortcut->inputs;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = l.n_delta + offset_o;
        float *out = shortcut->delta + offset_c;
        gradient_list_gpu(output, l.outputs, l.active);
        matrix_multiply_gpu(output, delta_n, l.inputs, delta_l);
        shortcut_gpu(delta_l, l.input_w, l.input_h, l.input_c, \
                     out, shortcut->input_w, shortcut->input_h, shortcut->input_c, \
                     1, 1, out);
    }
}
