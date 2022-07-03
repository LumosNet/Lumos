#include "mse_layer.h"

Layer *make_mse_layer(int group)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = MSE;
    l->group = group;

    l->forward = forward_mse_layer;
    l->backward = backward_mse_layer;

    l->update = NULL;
    l->init_layer_weights = NULL;

    fprintf(stderr, "Mse             Layer    :    [output=%4d]\n", 1);
    return l;
}

Layer *make_mse_layer_by_cfg(CFGParams *p)
{
    int group = 0;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "group")){
            group = atoi(param->val);
        }
        param = param->next;
    }

    Layer *l = make_mse_layer(group);
    return l;
}

void init_mse_layer(Layer *l, int w, int h, int c)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h*l->input_w*l->input_c;

    l->output_h = 1;
    l->output_w = 1;
    l->output_c = 1;
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->deltas = l->inputs;

    fprintf(stderr, "Mse             Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}


void forward_mse_layer(Layer l, int num)
{
    float *truth = calloc(l.group*num, sizeof(float));
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        int offset_t = i*l.group;
        float *input = l.input+offset_i;
        float *output = l.output+offset_o;
        float *label = truth+offset_t;
        one_hot_encoding(l.group, atoi(l.label[i]), label);
        subtract(label, input, l.inputs, l.workspace);
        gemm(1, 0, l.input_h, l.input_w, l.input_h, l.input_w, 1, \
            l.workspace, l.workspace, output);
        multy_cpu(output, l.outputs, 1/(float)l.group, 1);
    }
    free(truth);
}

void backward_mse_layer(Layer l, int num, float *n_delta)
{
    float *truth = calloc(l.group*num, sizeof(float));
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_t = i*l.group;
        float *input = l.input+offset_i;
        float *delta_l = l.delta+offset_i;
        float *label = truth+offset_t;
        one_hot_encoding(l.group, atoi(l.label[i]), label);
        subtract(input, label, l.inputs, delta_l);
        multy_cpu(delta_l, l.inputs, (float)2/l.group, 1);
    }
}