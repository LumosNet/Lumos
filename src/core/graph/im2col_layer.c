#include "im2col_layer.h"

Layer make_im2col_layer(CFGParams *p)
{
    Layer l = {0};
    l.type = IM2COL;
    l.weights = 0;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "flag")){
            l.im2col_flag = atoi(param->val);
        }
        param = param->next;
    }
    l.forward = forward_im2col_layer;
    l.backward = backward_im2col_layer;
    l.update = NULL;

    restore_im2col_layer(l);

    fprintf(stderr, "Im2col Layer: [flag=%d]\n", l.im2col_flag);
    return l;
}

void init_im2col_layer(Layer l, int w, int h, int c)
{
    l.input_h = h,
    l.input_w = w;
    l.input_c = c;
    l.inputs = l.input_h*l.input_w*l.input_c;

    l.output_h = 1;
    l.output_w = 1;
    l.output_c = 1;
    l.outputs = 1;

    l.workspace_size = 0;

    l.deltas = l.inputs;
}

void restore_im2col_layer(Layer l)
{
    l.input_h = -1;
    l.input_w = -1;
    l.input_c = -1;
    l.inputs = -1;

    l.output_h = -1;
    l.output_w = -1;
    l.output_c = -1;
    l.outputs = -1;

    l.workspace_size = -1;

    l.deltas = -1;

    l.input = NULL;
    l.output = NULL;
    l.delta = NULL;
}

void forward_im2col_layer(Layer l)
{
    memcpy(l.output, l.input, l.outputs*sizeof(float));
}

void backward_im2col_layer(Layer l, float *n_delta)
{
    memcpy(l.delta, n_delta, l.inputs*sizeof(float));
}
