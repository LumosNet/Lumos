#include "convolutional_layer.h"

void convolutional_layer(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        l->output[i] = convolutional(l->input[i], l->kernel_weights, l->pad, l->stride);
        if (l->bias){
            add_bias(l->output[i], l->bias_weights, l->filters, l->output_h*l->output_w);
        }
        activate_tensor(l->output[i], l->active);
    }
}

void backward_convolutional_layer(Layer *l, Network *net)
{
    return;
}