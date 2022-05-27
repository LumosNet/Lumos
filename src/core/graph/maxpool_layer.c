#include "maxpool_layer.h"

Layer make_maxpool_layer(CFGParams *p, int h, int w, int c)
{
    Layer l = {0};
    l.type = POOLING;
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;

    l.pad = 0;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "type")){
            if (0 == strcmp(param->val, "avg")) l.pool = AVG;
            else l.pool = MAX;
        } else if (0 == strcmp(param->key, "ksize")){
            l.ksize = atoi(param->val);
            l.stride = l.ksize;
        }
        param = param->next;
    }
    l.output_h = (l.input_h - l.ksize) / l.ksize + 1;
    l.output_w = (l.input_w - l.ksize) / l.ksize + 1;
    l.output_c = l.input_c;

    if (l.pool == AVG){
        l.forward = forward_avgpool_layer;
        l.backward = backward_avgpool_layer;
    }
    else{
        l.forward = forward_maxpool_layer;
        l.backward = backward_maxpool_layer;
    }

    l.workspace_size = l.output_h*l.output_w*l.ksize*l.ksize*l.output_c;

    int size_o = l.output_w * l.output_h * l.output_c;
    int size_d = l.input_w * l.input_h * l.input_c;
    l.output = calloc(batch*size_o, sizeof(float));
    l.delta = calloc(batch*size_d, sizeof(float));
    if (l.pool == MAX) l.index = malloc(batch*sizeof(int *));
    for (int i = 0; i < batch; ++i){
        if (l.pool == MAX) l.index[i] = calloc(l.output_h*l.output_w*l.output_c, sizeof(int));
    }

    l.inputs = l.input_c*l.input_h*l.input_w;
    l.outputs = l.output_c*l.output_h*l.output_w;

    char *type;
    if (l.pool == AVG) type = "avg";
    else type = "max";
    fprintf(stderr, "  %s              %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", \
            type, l.ksize, l.ksize, l.ksize, l.input_h, l.input_w, l.input_c, \
            l.output_h, l.output_w, l.output_c);
    return l;
}

void forward_maxpool_layer(Layer l, float *workspace)
{
    im2col(l.input+offset_i, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, net.workspace);
    for (int c = 0; c < l.input_c; ++c){
        for (int h = 0; h < l.output_h*l.output_w; h++){
            float max = -999;
            int max_index = -1;
            for (int w = 0; w < l.ksize*l.ksize; w++){
                int mindex = c*l.output_h*l.output_w*l.ksize*l.ksize + l.output_h*l.output_w*w + h;
                if (net.workspace[mindex] > max){
                    max = net.workspace[mindex];
                    max_index = (c*l.input_h*l.input_w)+(h/l.output_w*l.ksize+w/l.ksize)*l.input_w+(h%l.output_w*l.ksize+w%l.ksize);
                }
            }
            output[l.output_h*l.output_w*c + h] = max;
            l.index[i][l.output_h*l.output_w*c + h] = max_index;
        }
    }
}

void backward_maxpool_layer(Layer l, float *n_delta, float *workspace)
{
    for (int j = 0; j < l.output_h*l.output_w*l.output_c; ++j){
        l_delta[l.index[i][j]] = n_delta[j];
    }
}