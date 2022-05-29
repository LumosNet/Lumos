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

    return l;
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