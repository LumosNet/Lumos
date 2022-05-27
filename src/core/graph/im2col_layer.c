#include "im2col_layer.h"

Layer make_im2col_layer(CFGParams *p, int h, int w, int c)
{
    Layer l = {0};
    l.type = IM2COL;
    l.input_h = h,
    l.input_w = w;
    l.input_c = c;

    l.output_h = 1;
    l.output_w = 1;
    l.output_c = 1;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "flag")){
            int flag = atoi(param->val);
            if (flag) l.output_h = l.input_h*l.input_w*l.input_c;
            else l.output_w = l.input_h*l.input_w*l.input_c;
        }
        param = param->next;
    }
    l.inputs = l.input_c*l.input_h*l.input_w;
    l.outputs = l.output_c*l.output_h*l.output_w;
    l.forward = forward_im2col_layer;
    l.backward = backward_im2col_layer;
    l.output = calloc(l.outputs, sizeof(float));
    l.delta = calloc(l.inputs, sizeof(float));
    fprintf(stderr, "  im2col                      %4d x%4d x%4d   ->    %2d x%2d\n", \
            l.input_w, l.input_h, l.input_c, l.output_w, l.output_h);
    return l;
}

void forward_im2col_layer(Layer l, float *workspace)
{
    memcpy(l.output, l.input, l.outputs*sizeof(float));
}

void backward_im2col_layer(Layer l, float *n_delta, float *workspace)
{
    memcpy(l.delta, n_delta, n_delta*l.inputs*sizeof(float));
}