#include "layer.h"

Layer create_layer(CFGPiece *p)
{
    Layer layer;
    if (0 == strcmp(p->name, "convolutional")){
        layer = make_convolutional_layer(p);
    } else if (0 == strcmp(p->name, "pooling")){
        layer = make_pooling_layer(p);
    } else if (0 == strcmp(p->name, "softmax")){
        layer = make_softmax_layer(p);
    } else if (0 == strcmp(p->name, "connect")){
        layer = make_connect_layer(p);
    } else if (0 == strcmp(p->name, "im2col")){
        layer = make_im2col_layer(p);
    } else if (0 == strcmp(p->name, "mse")){
        layer = make_mse_layer(p);
    }
    return layer;
}
