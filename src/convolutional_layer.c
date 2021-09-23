#include "convolution.h"

void forward_convolutional_layer(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        for (int j = 0; j < l->filters; ++j){
            Image *channel = forward_convolutional(l->input[i], l->weights[j], l->pad, l->stride);
            memcpy(l->output[i]+j*l->width*l->height, channel, l->width*l->height);
            del(channel);
        }
        for (int k = 0; k < l->height*l->width*l->filters; ++k){
            Image *out = l->output[i];
            out->data[k] = l->active(out->data[k]);
        }
    }
}

void backward_convolutional_layer(Layer *l, Network *net)
{
    return;
}