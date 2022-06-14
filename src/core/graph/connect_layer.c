#include "connect_layer.h"

Layer *make_connect_layer(int output, int bias, char *active)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = CONNECT;
    l->bias = 1;
    l->filters = 1;
    l->weights = 1;

    l->ksize = output;

    l->active_str = active;
    Activation type = load_activate_type(active);
    l->active = load_activate(type);
    l->gradient = load_gradient(type);

    l->bias = bias;

    l->forward = forward_connect_layer;
    l->backward = backward_connect_layer;
    l->update = update_connect_layer;

    restore_connect_layer(l);

    fprintf(stderr, "Connect         Layer    :    [output=%4d, bias=%d, active=%s]\n", l->ksize, l->bias, l->active_str);
    return l;
}


Layer *make_connect_layer_by_cfg(CFGParams *p)
{
    int output = 0;
    int bias = 0;
    char *active = NULL;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "output")){
            output = atoi(param->val);
        } else if (0 == strcmp(param->key, "active")){
            active = param->val;
        } else if (0 == strcmp(param->key, "bias")){
            bias = atoi(param->val);
        }
        param = param->next;
    }

    Layer *l = make_connect_layer(output, bias, active);
    return l;
}

void init_connect_layer(Layer *l, int w, int h, int c)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h*l->input_w*l->input_c;

    l->output_h = l->ksize;
    l->output_w = 1;
    l->output_c = 1;
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->workspace_size = l->input_c*l->input_h*l->input_w*l->output_c*l->output_h*l->output_w;

    l->kernel_weights_size = l->input_h*l->output_h;
    l->bias_weights_size = l->output_h;
    l->deltas = l->inputs;
}

void restore_connect_layer(Layer *l)
{
    l->input_h = -1;
    l->input_w = -1;
    l->input_c = -1;
    l->inputs = -1;

    l->output_h = -1;
    l->output_w = -1;
    l->output_c = -1;
    l->outputs = -1;

    l->workspace_size = -1;

    l->kernel_weights_size = -1;
    l->bias_weights_size = -1;
    l->deltas = -1;

    l->input = NULL;
    l->output = NULL;
    l->kernel_weights = NULL;
    l->bias_weights = NULL;
    l->delta = NULL;
}

void forward_connect_layer(Layer l)
{
    gemm(0, 0, l.output_h, l.input_h, l.input_h, l.input_w, 
        1, l.kernel_weights, l.input, l.output);
    if (l.bias){
        add_bias(l.output, l.bias_weights, l.ksize, 1);
    }
    activate_list(l.output, l.outputs, l.active);
}

void backward_connect_layer(Layer l, float *n_delta)
{
    gradient_list(l.output, l.outputs, l.gradient);
    multiply(n_delta, l.output, l.outputs, n_delta);
    gemm(1, 0, l.output_h, l.input_h, l.output_h, l.output_w, 1, 
        l.kernel_weights, n_delta, l.delta);
}

void update_connect_layer(Layer l, float rate, float *n_delta)
{
    gemm(0, 1, l.output_h, l.output_w, \
        l.input_h, l.input_w, 1, \
        n_delta, l.input, l.workspace);
    saxpy(l.update_kernel_weights, l.workspace, l.output_h * l.input_h, rate, l.update_kernel_weights);
    if (l.bias){
        saxpy(l.update_bias_weights, n_delta, l.outputs, rate, l.update_bias_weights);
    }
}
