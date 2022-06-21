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

    restore_im2col_layer(l);

    fprintf(stderr, "Im2col          Layer    :    [flag=%d]\n", l->im2col_flag);
    return l;
}

Layer *make_im2col_layer_by_cfg(CFGParams *p)
{
    int flag = 1;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "flag")){
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
    l->inputs = l->input_h*l->input_w*l->input_c;

    l->output_h = 1;
    l->output_w = 1;
    l->output_c = 1;
    if (l->im2col_flag) l->output_h = l->inputs;
    else l->output_w = l->inputs;
    l->outputs = l->inputs;

    l->workspace_size = 0;

    l->deltas = l->inputs;

    fprintf(stderr, "Im2col          Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void restore_im2col_layer(Layer *l)
{
    l->input_h = -1;
    l->input_w = -1;
    l->input_c = -1;
    l->inputs = -1;

    l->output_h = -1;
    l->output_w = -1;
    l->output_c = -1;
    l->outputs = -1;

    l->workspace_size = -1;

    l->deltas = -1;

    l->input = NULL;
    l->output = NULL;
    l->delta = NULL;
}

void forward_im2col_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int input_offset = i*l.inputs;
        int output_offset = i*l.outputs;
        float *input = l.input+input_offset;
        float *output = l.output+output_offset;
        memcpy(output, input, l.outputs*sizeof(float));
    }
}

void backward_im2col_layer(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        if (n_delta){
            memcpy(l.delta+offset_i, n_delta+offset_o, l.inputs*sizeof(float));
        }
    }
}
