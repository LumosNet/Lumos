#include "im2col_layer.h"

Layer *make_im2col_layer()
{
    Layer *l = malloc(sizeof(Layer));
    l->type = IM2COL;

    l->initialize = init_im2col_layer;
    l->forward = forward_im2col_layer;
    l->backward = backward_im2col_layer;

    l->initializegpu = init_im2col_layer_gpu;
    l->forwardgpu = forward_im2col_layer_gpu;
    l->backwardgpu = backward_im2col_layer_gpu;

    l->weightinit = NULL;
    l->weightinitgpu = NULL;

    l->update = NULL;
    l->updategpu = NULL;

    fprintf(stderr, "Im2col          Layer\n");
    return l;
}

void init_im2col_layer(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h,
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = 1;
    l->output_w = 1;
    l->output_c = l->inputs;
    l->outputs = l->inputs;

    l->workspace_size = 0;
    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));

    fprintf(stderr, "Im2col          Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_im2col_layer(Layer l, int num)
{
    memcpy(l.output, l.input, num*l.outputs*sizeof(float));
}

void backward_im2col_layer(Layer l, float rate, int num, float *n_delta)
{
    memcpy(l.delta, n_delta, num*l.inputs*sizeof(float));
}
