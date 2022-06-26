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

    l->init_layer_weights = init_connect_weights;

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

    l->kernel_weights_size = l->inputs*l->outputs;
    l->bias_weights_size = l->outputs;
    l->deltas = l->inputs;

    fprintf(stderr, "Connect         Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
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

void init_connect_weights(Layer *l)
{
    random(1, l->inputs, 0.01, l->kernel_weights_size, l->kernel_weights);
    for (int i = 0; i < l->bias_weights_size; ++i){
        l->bias_weights[i] = 0.01;
    }
}

void forward_connect_layer(Layer l, int num)
{
    fill_cpu(l.delta, l.deltas*num, 0, 1);
    for (int i = 0; i < num; ++i){
        int input_offset = i*l.inputs;
        int output_offset = i*l.outputs;
        float *input = l.input+input_offset;
        float *output = l.output+output_offset;
        gemm(0, 0, l.outputs, l.inputs, l.inputs, 1, 
            1, l.kernel_weights, input, output);
        if (l.bias){
            add_bias(output, l.bias_weights, l.ksize, 1);
        }
        printf("\n------------------output------------------\n");
        for (int j = 0; j < l.outputs; ++j){
            printf("%f ", output[j]);
        }
        printf("\n------------------output------------------\n");
        activate_list(output, l.outputs, l.active);
    }
    printf("finish connect\n");
}

void backward_connect_layer(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        float *output = l.output+offset_o;
        float *delta_l = l.delta+offset_i;
        float *delta_n = n_delta+offset_o;
        gradient_list(output, l.outputs, l.gradient);
        multiply(delta_n, output, l.outputs, delta_n);
        gemm(1, 0, l.output_h, l.input_h, l.output_h, l.input_w, 1, 
            l.kernel_weights, delta_n, delta_l);
    }
}

void update_connect_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        float *input = l.input+offset_i;
        float *delta_n = n_delta+offset_o;
        gemm(0, 1, l.output_h, l.output_w, \
            l.input_h, l.input_w, 1, \
            delta_n, input, l.workspace);
        saxpy(l.update_kernel_weights, l.workspace, l.output_h * l.input_h, rate, l.update_kernel_weights);
        if (l.bias){
            saxpy(l.update_bias_weights, delta_n, l.outputs, rate, l.update_bias_weights);
        }
    }
}
