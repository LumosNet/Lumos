#include "mse_layer.h"

Layer make_mse_layer(LayerParams *p, int batch, int h, int w, int c)
{
    Layer l = {0};
    l.type = MSE;
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;
    Node *n = p->head;
    while (n){
        Params *param = n->val;
        if (0 == strcmp(param->key, "group")){
            l.group = atoi(param->val);
        } else if (0 == strcmp(param->key, "noloss")){
            l.noloss = atoi(param->val);
        }
        n = n->next;
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

    fprintf(stderr, "  mse              %5d      %4d x%4d         ->  %4d x%4d\n", \
            l.group, l.input_w, l.input_h, l.output_w, l.output_h);
    return l;
}

void forward_mse_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        debug_data(net.fdebug, l.input_h, l.input_w, net.labels[i].data, "\nlabels\n");
        debug_data(net.fdebug, l.input_h, l.input_w, l.input+i*l.inputs, "\nmse_layer input\n");
        subtract(net.labels[i].data, l.input+i*l.inputs, l.inputs, net.workspace);
        // debug_data(net.fdebug, l.input_h, l.input_w, net.workspace, "\nmse_layer substruct\n");
        gemm(1, 0, l.input_h, l.input_w, l.input_h, l.input_w, 1, \
             net.workspace, net.workspace, l.output+i*l.outputs);
        // debug_data(net.fdebug, l.output_h, l.output_w, l.output+i*l.outputs, "\nmse_layer output\n");
        l.output[i*l.outputs] /= l.group;
        // printf("label: %f\n", l.input[i*l.inputs]);
    }
}

void backward_mse_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        subtract(l.input+i*l.inputs, net.labels[i].data, l.inputs, l.delta+i*l.inputs);
        // debug_data(net.bdebug, l.input_h, l.input_w, l.delta+i*l.inputs, "\nmse_layer delta\n");
    }
}