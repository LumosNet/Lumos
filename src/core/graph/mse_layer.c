#include "mse_layer.h"

Layer make_mse_layer(CFGParams *p)
{
    Layer l = {0};
    l.type = MSE;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "group")){
            l.group = atoi(param->val);
        }
        param = param->next;
    }

    l.forward = forward_mse_layer;
    l.backward = backward_mse_layer;
    l.update = NULL;

    restore_mse_layer(l);

    return l;
}

void init_mse_layer(Layer l, int w, int h, int c)
{
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;
    l.inputs = l.input_h*l.input_w*l.input_c;

    l.output_h = 1;
    l.output_w = 1;
    l.output_c = 1;
    l.outputs = l.output_h*l.output_w*l.output_c;

    l.deltas = l.inputs;
}

void restore_mse_layer(Layer l)
{
    l.input_h = -1;
    l.input_w = -1;
    l.input_c = -1;
    l.inputs = -1;

    l.output_h = -1;
    l.output_w = -1;
    l.output_c = -1;
    l.outputs = -1;

    l.workspace_size = -1;

    l.deltas = -1;

    l.input = NULL;
    l.output = NULL;
    l.delta = NULL;
}

// void forward_mse_layer(Layer l, float *workspace)
// {
//     one_hot_encoding(l.group, net.labels[i].data[0], l.truth+i*l.group);
//     subtract(l.truth+i*l.group, l.input+i*l.inputs, l.inputs, net.workspace);
//     gemm(1, 0, l.input_h, l.input_w, l.input_h, l.input_w, 1, \
//         net.workspace, net.workspace, l.output+i*l.outputs);
//     l.output[i*l.outputs] /= l.group;
// }

// void backward_mse_layer(Layer l, Network net)
// {
//     subtract(l.input+i*l.inputs, l.truth+i*l.group, l.inputs, l.delta+i*l.inputs);
//     multy_cpu(l.delta+i*l.inputs, l.inputs, (float)2/l.group, 1);
// }