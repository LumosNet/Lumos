#include "pooling_layer.h"

PoolingType load_pooling_type(char *pool)
{
    if (0 == strcmp(pool, "avg")) return AVG;
    return MAX;
}

Layer make_pooling_layer(LayerParams *p, int batch, int h, int w, int c)
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

    char *type;
    if (l.pool == AVG) type = "avg";
    else type = "max";
    fprintf(stderr, "  %s              %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", \
            type, l.ksize, l.ksize, l.ksize, l.input_h, l.input_w, l.input_c, \
            l.output_h, l.output_w, l.output_c);
    return l;
}