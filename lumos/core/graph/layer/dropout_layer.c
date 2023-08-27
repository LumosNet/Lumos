#include "dropout_layer.h"

Layer *make_dropout_layer(float probability)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = DROPOUT;
    l->probability = probability;
    l->update = NULL;

    l->initialize = init_dropout_layer;
    l->forward = forward_dropout_layer;
    l->backward = backward_dropout_layer;
    l->initialize_gpu = init_dropout_layer_gpu;
    l->forward_gpu = forward_dropout_layer_gpu;
    l->backward_gpu = backward_dropout_layer_gpu;

    fprintf(stderr, "Dropout   Layer    :    [probability=%.2f]\n", l->probability);
    return l;
}

void init_dropout_layer(Layer *l, int w, int h, int c)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = h;
    l->output_w = w;
    l->output_c = c;
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->workspace_size = l->inputs;

    l->output = calloc(l->outputs*l->subdivision, sizeof(float));
    l->delta = calloc(l->inputs*l->subdivision, sizeof(float));

    fprintf(stderr, "Dropout         Layer\n");
}

void forward_dropout_layer(Layer l, int num)
{
    if (!l.train){
        memcpy(l.output, l.input, num*l.inputs*sizeof(float));
        return;
    }
    float scale = 1. / (1.-l.probability);
    for (int i = 0; i < num*l.inputs; ++i){
        float r = rand_uniform(0, 1);
        l.dropout_rand[i] = r;
        if (r < l.probability) l.output[i] = 0;
        else l.output[i] = l.input[i] * scale;
    }
}

void backward_dropout_layer(Layer l, float rate, int num)
{
    if (!l.train){
        memcpy(l.delta, l.n_delta, num*l.inputs*sizeof(float));
        return;
    }
    float scale = 1. / (1.-l.probability);
    for (int i = 0; i < num*l.inputs; ++i){
        float r = l.dropout_rand[i];
        if (r < l.probability) l.delta[i] = 0;
        else l.delta[i] = n_delta[i] * scale;
    }
}
