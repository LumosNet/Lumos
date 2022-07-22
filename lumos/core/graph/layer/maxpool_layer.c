#include "maxpool_layer.h"

Layer *make_maxpool_layer(int ksize)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = MAXPOOL;
    l->pad = 0;
    l->weights = 0;

    l->ksize = ksize;
    l->stride = ksize;

    l->forward = forward_maxpool_layer;
    l->backward = backward_maxpool_layer;

    l->update = NULL;
    l->init_layer_weights = NULL;

    fprintf(stderr, "Max Pooling     Layer    :    [ksize=%2d]\n", l->ksize);
    return l;
}

Layer *make_maxpool_layer_by_cfg(CFGParams *p)
{
    int ksize = 0;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "ksize")){
            ksize = atoi(param->val);
        }
        param = param->next;
    }

    Layer *l = make_maxpool_layer(ksize);
    return l;
}

void init_maxpool_layer(Layer *l, int w, int h, int c)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h*l->input_w*l->input_c;

    l->output_h = (l->input_h - l->ksize) / l->ksize + 1;
    l->output_w = (l->input_w - l->ksize) / l->ksize + 1;
    l->output_c = l->input_c;
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->workspace_size = l->output_h*l->output_w*l->ksize*l->ksize*l->output_c;

    l->deltas = l->inputs;

    fprintf(stderr, "Max Pooling     Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_maxpool_layer(Layer l, int num)
{
    fill_cpu(l.delta, l.deltas*num, 0, 1);
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        float *input = l.input+offset_i;
        float *output = l.output+offset_o;
        im2col(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
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
                output[l.output_h*l.output_w*c + h] = max;
                l.maxpool_index[l.output_h*l.output_w*c + h] = max_index;
            }
        }
    }
}

void backward_maxpool_layer(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        float *delta_l = l.delta+offset_i;
        float *delta_n = n_delta+offset_o;
        for (int j = 0; j < l.output_h*l.output_w*l.output_c; ++j){
            delta_l[l.maxpool_index[j]] = delta_n[j];
        }
    }
}
