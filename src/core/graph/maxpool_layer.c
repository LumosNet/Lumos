#include "maxpool_layer.h"

Layer make_maxpool_layer(CFGParams *p)
{
    Layer l = {0};
    l.type = MAXPOOL;
    l.pad = 0;
    l.weights = 0;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "ksize")){
            l.ksize = atoi(param->val);
            l.stride = l.ksize;
        }
        param = param->next;
    }

    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update = NULL;

    restore_maxpool_layer(l);

    return l;
}

void init_maxpool_layer(Layer l, int w, int h, int c)
{
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;
    l.inputs = l.input_h*l.input_w*l.input_c;

    l.output_h = (l.input_h - l.ksize) / l.ksize + 1;
    l.output_w = (l.input_w - l.ksize) / l.ksize + 1;
    l.output_c = l.input_c;
    l.outputs = l.output_h*l.output_w*l.output_c;

    l.workspace_size = l.output_h*l.output_w*l.ksize*l.ksize*l.output_c;

    l.deltas = l.inputs;
}

void restore_maxpool_layer(Layer l)
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

void forward_maxpool_layer(Layer l)
{
    im2col(l.input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
    for (int c = 0; c < l.input_c; ++c){
        for (int h = 0; h < l.output_h*l.output_w; h++){
            float max = -999;
            int max_index = -1;
            for (int w = 0; w < l.ksize*l.ksize; w++){
                int mindex = c*l.output_h*l.output_w*l.ksize*l.ksize + l.output_h*l.output_w*w + h;
                if (l.workspace[mindex] > max){
                    max = l.workspace[mindex];
                    max_index = (c*l.input_h*l.input_w)+(h/l.output_w*l.ksize+w/l.ksize)*l.input_w+(h%l.output_w*l.ksize+w%l.ksize);
                }
            }
            l.output[l.output_h*l.output_w*c + h] = max;
            l.maxpool_index[l.output_h*l.output_w*c + h] = max_index;
        }
    }
}

void backward_maxpool_layer(Layer l, float *n_delta)
{
    for (int j = 0; j < l.output_h*l.output_w*l.output_c; ++j){
        l.delta[l.maxpool_index[j]] = n_delta[j];
    }
}
