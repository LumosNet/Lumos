#include "softmax_layer.h"

void forward_softmax_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        float *output = l.output+offset_o;
        float *input = l.input+offset_i;
        float sum = 0;
        for (int j = 0; j < l.group; ++j){
            net.workspace[j] = exp(input[j]);
            sum += net.workspace[j];
        }
        for (int j = 0; j < l.group; ++j){
            output[j] = net.workspace[j] / sum;
        }
    }
}

void backward_softmax_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        float *input = l.input + offset_i;
        float *delta = l.delta+offset_i;
        float sum = 0;
        for (int j = 0; j < l.group; ++j){
            net.workspace[j] = exp(input[j]);
            sum += net.workspace[j];
        }
        for (int j = 0; j < l.group; ++j){
            delta[j] = (1 + (input[j]*net.workspace[j]) / sum) / sum;
        }
    }
}

Layer make_softmax_layer(LayerParams *p, int batch, int h, int w, int c)
{
    Layer l = {0};
    l.type = SOFTMAX;
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;
    Node *n = p->head;
    while (n){
        Params *param = n->val;
        if (0 == strcmp(param->key, "group")){
            l.group = atoi(param->val);
        }
        n = n->next;
    }
    l.output_h = l.group;
    l.output_w = 1;
    l.output_c = 1;
    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;

    l.workspace_size = l.group*l.group;

    int size_o = l.output_w * l.output_h * l.output_c;
    int size_d = l.input_w * l.input_h * l.input_c;
    l.output = calloc(batch*size_o, sizeof(float));
    l.delta = calloc(batch*size_d, sizeof(float));

    fprintf(stderr, "  softmax          %5d      %4d x%4d         ->  %4d x%4d\n", \
            l.group, l.input_w, l.input_h, l.output_w, l.output_h);
    return l;
}