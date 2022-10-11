#include "im2col_layer.h"

Layer *make_im2col_layer(int flag)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = IM2COL;
    l->weights = 0;

    l->im2col_flag = flag;

    l->forward = forward_im2col_layer;
    l->backward = backward_im2col_layer;

    l->update = NULL;
    l->init_layer_weights = NULL;

    fprintf(stderr, "Im2col          Layer    :    [flag=%d]\n", l->im2col_flag);
    return l;
}

Layer *make_im2col_layer_by_cfg(CFGParams *p)
{
    int flag = 1;

    CFGParam *param = p->head;
    while (param)
    {
        if (0 == strcmp(param->key, "flag"))
        {
            flag = atoi(param->val);
        }
        param = param->next;
    }

    Layer *l = make_im2col_layer(flag);
    return l;
}

void init_im2col_layer(Layer *l, int w, int h, int c)
{
    l->input_h = h,
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = 1;
    l->output_w = 1;
    l->output_c = 1;
    if (l->im2col_flag)
        l->output_h = l->inputs;
    else
        l->output_w = l->inputs;
    l->outputs = l->inputs;

    l->workspace_size = 0;

    l->deltas = l->inputs;

    fprintf(stderr, "Im2col          Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_im2col_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        memcpy(output, input, l.outputs * sizeof(float));
    }
}

void backward_im2col_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        memcpy(delta_l, delta_n, l.inputs * sizeof(float));
    }
}
