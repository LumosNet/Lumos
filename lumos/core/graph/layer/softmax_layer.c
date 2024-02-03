#include "softmax_layer.h"

Layer *make_softmax_layer(int group)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = SOFTMAX;
    l->group = group;

    l->initialize = init_softmax_layer;
    l->forward = forward_softmax_layer;
    l->backward = backward_softmax_layer;

    l->initializegpu = init_softmax_layer_gpu;
    l->forwardgpu = forward_softmax_layer_gpu;
    l->backwardgpu = backward_softmax_layer_gpu;

    l->weightinit = NULL;
    l->weightinitgpu = NULL;

    l->update = NULL;
    l->updategpu = NULL;

    fprintf(stderr, "Softmax         Layer    :    [output=%4d]\n", group);
    return l;
}

void init_softmax_layer(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h*l->input_w*l->input_c;

    l->output_h = h;
    l->output_w = w;
    l->output_c = c;
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->workspace_size = l->inputs+1;

    l->output = calloc(l->outputs*subdivision, sizeof(float));
    l->delta = calloc(l->inputs*subdivision, sizeof(float));

    fprintf(stderr, "Softmax         Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_softmax_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        softmax(input, l.inputs, output);
    }
}

void backward_softmax_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        float *input = l.input + offset_i;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        softmax_grident(input, l.inputs, delta_l);
        matrix_multiply_cpu(delta_n, delta_l, l.inputs, delta_l);
    }
}
