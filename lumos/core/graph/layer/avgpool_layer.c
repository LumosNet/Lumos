#include "avgpool_layer.h"

Layer *make_avgpool_layer(int ksize)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = AVGPOOL;
    l->pad = 0;
    l->weights = 0;

    l->ksize = ksize;
    l->stride = l->ksize;

    l->forward = forward_avgpool_layer;
    l->backward = backward_avgpool_layer;

    l->update = NULL;
    l->init_layer_weights = NULL;

    fprintf(stderr, "Avg Pooling     Layer    :    [ksize=%2d]\n", l->ksize);
    return l;
}

Layer *make_avgpool_layer_by_cfg(CFGParams *p)
{
    CFGParam *param = p->head;
    int ksize = 0;
    while (param)
    {
        if (0 == strcmp(param->key, "ksize"))
        {
            ksize = atoi(param->val);
        }
        param = param->next;
    }
    Layer *l = make_avgpool_layer(ksize);

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
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        im2col(input, l.input_h, l.input_w, l.input_c,
               l.ksize, l.stride, l.pad, l.workspace);
        for (int c = 0; c < l.output_c; ++c)
        {
            for (int h = 0; h < l.output_h; ++h)
            {
                for (int w = 0; w < l.output_w; ++w)
                {
                    for (int k = 0; k < l.ksize * l.ksize; ++k)
                    {
                        output[c * l.output_h * l.output_w + h * l.output_w + w] +=
                            l.workspace[(c * l.ksize * l.ksize + k) * l.output_h * l.output_w + h * l.output_w + w] * (float)(1 / (float)(l.ksize * l.ksize));
                    }
                }
            }
        }
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
        for (int c = 0; c < l.input_c; ++c)
        {
            for (int h = 0; h < l.input_h; ++h)
            {
                for (int w = 0; w < l.input_w; ++w)
                {
                    int height_index = h / l.ksize;
                    int width_index = w / l.ksize;
                    delta_l[l.input_h * l.input_w * c + l.input_w * h + w] =
                        delta_n[c * l.output_h * l.output_w + height_index * l.output_w + width_index] * (float)(1 / (float)(l.ksize * l.ksize));
                }
            }
        }
    }
}
