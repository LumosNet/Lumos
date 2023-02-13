#include "avgpool_layer.h"

Layer *make_avgpool_layer(int ksize)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = AVGPOOL;
    l->pad = 0;
    l->weights = 0;

    l->ksize = ksize;
    l->stride = l->ksize;
#ifdef GPU
    l->forward = forward_avgpool_layer_gpu;
    l->backward = backward_avgpool_layer_gpu;
#else
    l->forward = forward_avgpool_layer;
    l->backward = backward_avgpool_layer;
#endif
    l->update = NULL;
    l->init_layer_weights = NULL;

    fprintf(stderr, "Avg Pooling     Layer    :    [ksize=%2d]\n", l->ksize);
    return l;
}

void init_avgpool_layer(Layer *l, int w, int h, int c)
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
    fprintf(stderr, "Avg Pooling     Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_avgpool_layer(Layer l, int num)
{
    fill_cpu(l.output, l.outputs * num, 0, 1);
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        avgpool(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, output);
    }
}

void backward_avgpool_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        avgpool_gradient(delta_l, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, delta_n);
    }
}
