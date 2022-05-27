#include "mse_layer.h"

Layer make_mse_layer(CFGParams *p, int h, int w, int c)
{
    Layer l = {0};
    l.type = MSE;
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "group")){
            l.group = atoi(param->val);
        } else if (0 == strcmp(param->key, "noloss")){
            l.noloss = atoi(param->val);
        }
        param = param->next;
    }
    l.output_h = 1;
    l.output_w = 1;
    l.output_c = 1;

    l.forward = forward_mse_layer;
    l.backward = backward_mse_layer;

    l.workspace_size = l.group;

    l.inputs = l.input_c*l.input_h*l.input_w;
    l.outputs = l.output_c*l.output_h*l.output_w;
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.inputs, sizeof(float));

    l.truth = calloc(batch*l.group, sizeof(float));

    fprintf(stderr, "  mse              %5d      %4d x%4d         ->  %4d x%4d\n", \
            l.group, l.input_w, l.input_h, l.output_w, l.output_h);
    return l;
}

void forward_mse_layer(Layer l, float *workspace)
{
    one_hot_encoding(l.group, net.labels[i].data[0], l.truth+i*l.group);
    subtract(l.truth+i*l.group, l.input+i*l.inputs, l.inputs, net.workspace);
    gemm(1, 0, l.input_h, l.input_w, l.input_h, l.input_w, 1, \
        net.workspace, net.workspace, l.output+i*l.outputs);
    l.output[i*l.outputs] /= l.group;
}

void backward_mse_layer(Layer l, Network net)
{
    subtract(l.input+i*l.inputs, l.truth+i*l.group, l.inputs, l.delta+i*l.inputs);
    multy_cpu(l.delta+i*l.inputs, l.inputs, (float)2/l.group, 1);
}