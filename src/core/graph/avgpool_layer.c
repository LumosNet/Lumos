#include "avgpool_layer.h"

Layer make_avgpool_layer(CFGParams *p)
{
    Layer l = {0};
    l.type = AVGPOOL;
    l.pad = 0;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "ksize")){
            l.ksize = atoi(param->val);
            l.stride = l.ksize;
        }
        param = param->next;
    }

    l.forward = forward_avgpool_layer;
    l.backward = backward_avgpool_layer;
    l.update = NULL;

    return l;
}

void forward_avgpool_layer(Layer l)
{
    im2col(l.input, l.input_h, l.input_w, l.input_c,
        l.ksize, l.stride, l.pad, l.workspace);
    for (int c = 0; c < l.output_c; ++c){
        for (int h = 0; h < l.output_h; ++h){
            for (int w = 0; w < l.output_w; ++w){
                for (int k = 0; k < l.ksize*l.ksize; ++k){
                    l.output[c*l.output_h*l.output_w + h*l.output_w + w] +=
                    l.workspace[(c*l.ksize*l.ksize+k)*l.output_h*l.output_w+h*l.output_w+w] * (float)(1 / (float)(l.ksize*l.ksize));
                }
            }
        }
    }
}

void backward_avgpool_layer(Layer l, float *n_delta)
{
    for (int c = 0; c < l.input_c; ++c){
        for (int h = 0; h < l.input_h; ++h){
            for (int w = 0; w < l.input_w; ++w){
                int height_index = h / l.ksize;
                int width_index = w / l.ksize;
                l.delta[l.input_h*l.input_w*c + l.input_w*h + w] =
                n_delta[c*l.output_h*l.output_w + height_index*l.output_w + width_index] * (float)(1 / (float)(l.ksize*l.ksize));
            }
        }
    }
}