#include "maxpool_layer.h"

Layer *make_maxpool_layer(int ksize)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = MAXPOOL;
    l->pad = 0;
    l->weights = 0;

    l->ksize = ksize;
    l->stride = ksize;

#ifdef GPU
    l->forward = forward_maxpool_layer_gpu;
    l->backward = backward_maxpool_layer_gpu;
#else
    l->forward = forward_maxpool_layer;
    l->backward = backward_maxpool_layer;
#endif

    l->update = NULL;
    l->init_layer_weights = NULL;

    fprintf(stderr, "Max Pooling     Layer    :    [ksize=%2d]\n", l->ksize);
    return l;
}

void init_maxpool_layer(Layer *l, int w, int h, int c)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = (l->input_h - l->ksize) / l->ksize + 1;
    l->output_w = (l->input_w - l->ksize) / l->ksize + 1;
    l->output_c = l->input_c;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = l->output_h * l->output_w * l->ksize * l->ksize * l->output_c;

    l->deltas = l->inputs;

    fprintf(stderr, "Max Pooling     Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_maxpool_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        int *index = l.maxpool_index + offset_o;
        maxpool(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, output, index);
    }
}

void backward_maxpool_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        int *index = l.maxpool_index + offset_o;
        maxpool_gradient(delta_l, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, delta_n, index);
    }
}
