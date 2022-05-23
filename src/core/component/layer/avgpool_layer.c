#include "avgpool_layer.h"

Layer make_avgpool_layer(LayerParams *p, int batch, int h, int w, int c)
{
    Layer l = {0};
    l.type = POOLING;
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;

    l.pad = 0;
    Node *n = p->head;
    while (n){
        Params *param = n->val;
        if (0 == strcmp(param->key, "type")){
            if (0 == strcmp(param->val, "avg")) l.pool = AVG;
            else l.pool = MAX;
        } else if (0 == strcmp(param->key, "ksize")){
            l.ksize = atoi(param->val);
            l.stride = l.ksize;
        }
        n = n->next;
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

void forward_avgpool_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        im2col(l.input+offset_i, l.input_h, l.input_w, l.input_c, 
            l.ksize, l.stride, l.pad, net.workspace);
        for (int c = 0; c < l.output_c; ++c){
            for (int h = 0; h < l.output_h; ++h){
                for (int w = 0; w < l.output_w; ++w){
                    for (int k = 0; k < l.ksize*l.ksize; ++k){
                        l.output[offset_o+c*l.output_h*l.output_w + h*l.output_w + w] += 
                        net.workspace[(c*l.ksize*l.ksize+k)*l.output_h*l.output_w+h*l.output_w+w] * (float)(1 / (float)(l.ksize*l.ksize));
                    }
                }
            }
        }
    }
}

void backward_avgpool_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_ld = i*l.inputs;
        int offset_nd = i*l.outputs;
        for (int c = 0; c < l.input_c; ++c){
            for (int h = 0; h < l.input_h; ++h){
                for (int w = 0; w < l.input_w; ++w){
                    int height_index = h / l.ksize;
                    int width_index = w / l.ksize;
                    l.delta[offset_ld + l.input_h*l.input_w*c + l.input_w*h + w] = 
                    net.delta[offset_nd + c*l.output_h*l.output_w + height_index*l.output_w + width_index] * (float)(1 / (float)(l.ksize*l.ksize));
                }
            }
        }
    }
}