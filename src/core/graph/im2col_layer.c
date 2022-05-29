#include "im2col_layer.h"

Layer make_im2col_layer(CFGParams *p)
{
    Layer l = {0};
    l.type = IM2COL;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "flag")){
            int flag = atoi(param->val);
            if (flag) l.output_h = l.input_h*l.input_w*l.input_c;
            else l.output_w = l.input_h*l.input_w*l.input_c;
        }
        param = param->next;
    }
    l.forward = forward_im2col_layer;
    l.backward = backward_im2col_layer;
    return l;
}

void forward_im2col_layer(Layer l)
{
    memcpy(l.output, l.input, l.outputs*sizeof(float));
}

void backward_im2col_layer(Layer l, float *n_delta)
{
    memcpy(l.delta, n_delta, l.inputs*sizeof(float));
}
