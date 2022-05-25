#include "layer.h"

Layer create_layer(Network *net, LayerParams *p, int h, int w, int c)
{
    Layer layer;
    if (0 == strcmp(p->type, "convolutional")){
        layer = make_convolutional_layer(p, net->batch, h, w, c);
    } else if (0 == strcmp(p->type, "pooling")){
        layer = make_pooling_layer(p, net->batch, h, w, c);
    } else if (0 == strcmp(p->type, "softmax")){
        layer = make_softmax_layer(p, net->batch, h, w, c);
    } else if (0 == strcmp(p->type, "connect")){
        layer = make_connect_layer(p, net->batch, h, w, c);
    } else if (0 == strcmp(p->type, "im2col")){
        layer = make_im2col_layer(p, net->batch, h, w, c);
    } else if (0 == strcmp(p->type, "mse")){
        layer = make_mse_layer(p, net->batch, h, w, c);
    }
    if (layer.workspace_size > net->workspace_size) net->workspace_size = layer.workspace_size;
    return layer;
}
