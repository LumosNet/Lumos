#include "avgpool_layer.h"

Layer *make_avgpool_layer(int ksize, int stride, int pad)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = AVGPOOL;
    l->pad = pad;

    l->ksize = ksize;
    l->stride = stride;

    l->initialize = init_avgpool_layer;
    l->forward = forward_avgpool_layer;
    l->backward = backward_avgpool_layer;

    l->initializegpu = init_avgpool_layer_gpu;
    l->forwardgpu = forward_avgpool_layer_gpu;
    l->backwardgpu = backward_avgpool_layer_gpu;

    l->weightinit = NULL;
    l->weightinitgpu = NULL;

    l->update = NULL;
    l->updategpu = NULL;

    fprintf(stderr, "Avg Pooling     Layer    :    [ksize=%2d]\n", l->ksize);
    return l;
}

void init_avgpool_layer(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = (h + 2 * l->pad - l->ksize) / l->stride + 1;
    l->output_w = (w + 2 * l->pad - l->ksize) / l->stride + 1;
    l->output_c = l->input_c;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = 0;

    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));

    fprintf(stderr, "Avg Pooling     Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_avgpool_layer(Layer l, int num)
{
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
