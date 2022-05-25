#include "connect_layer.h"

Layer make_connect_layer(CFGParams *p, int h, int w, int c)
{
    Layer l = {0};
    l.type = CONNECT;
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;
    l.bias = 1;
    l.filters = 1;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "output")){
            l.ksize = atoi(param->val);
        } else if (0 == strcmp(param->key, "active")){
            Activation type = load_activate_type(param->val);
            l.active = load_activate(type);
            l.gradient = load_gradient(type);
        } else if (0 == strcmp(param->key, "bias")){
            l.bias = atoi(param->val);
        }
        param = param->next;
    }
    l.output_h = l.ksize;
    l.output_w = 1;
    l.output_c = 1;

    l.inputs = l.input_c*l.input_h*l.input_w;
    l.outputs = l.output_c*l.output_h*l.output_w;

    l.forward = forward_connect_layer;
    l.backward = backward_connect_layer;
    l.lweights = load_connect_weights;
    l.sweights = save_connect_weights;
    l.update = update_connect_layer;

    l.workspace_size = l.input_c*l.input_h*l.input_w*l.output_c*l.output_h*l.output_w;

    int size_k = l.input_h * l.output_h;
    l.kernel_weights = calloc(size_k, sizeof(float));
    if (l.bias) l.bias_weights = calloc(l.output_h, sizeof(float));

    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.inputs, sizeof(float));

    fprintf(stderr, "  connect             %2d      %4d x%4d         ->  %4d x%4d\n", \
            l.ksize, 1, h*w*c, l.output_w, l.output_h);
    return l;
}

void forward_connect_layer(Layer l, float *workspace)
{
    gemm(0, 0, l.output_h, l.input_h, l.input_h, l.input_w, 
        1, l.kernel_weights, l.input, l.output);
    if (l.bias){
        add_bias(l.output, l.bias_weights, l.ksize, 1);
    }
    activate_list(l.output, l.outputs, l.active);
}

void backward_connect_layer(Layer l, float *n_delta, float *workspace)
{
    gradient_list(l.output, l.outputs, l.gradient);
    multiply(net.delta, l.output, l.outputs, net.delta);
    gemm(1, 0, l.output_h, l.input_h, l.output_h, l.output_w, 1, 
        l.kernel_weights, net.delta, l.delta);
}

void update_connect_layer(Layer l, float rate, float *n_delta, float *workspace)
{
    gemm(0, 1, l.output_h, l.output_w, \
        l.input_h, l.input_w, 1, \
        net.delta, l.input, net.workspace);
    saxpy(l.kernel_weights, net.workspace, l.output_h * l.input_h, rate, l.kernel_weights);
    if (l.bias){
        saxpy(l.bias_weights, net.delta, l.outputs, rate, l.bias_weights);
    }
}

void save_connect_weights(Layer l, FILE *file)
{
    fwrite(l.kernel_weights, sizeof(float), l.output_h * l.input_h, file);
    if (l.bias) fwrite(l.bias_weights, sizeof(float), l.output_h, file);
}

void load_connect_weights(Layer l, FILE *file)
{
    if (file){
        fread(l.kernel_weights, sizeof(float), l.output_h * l.input_h, file);
        fread(l.bias_weights, sizeof(float), l.output_h, file);
    } else{
        for (int i = 0; i < l.output_h*l.input_h; ++i){
            l.kernel_weights[i] = 2.0*rand()/RAND_MAX-1;
        }
        if (l.bias){
            for (int i = 0; i < l.output_h; ++i){
                l.bias_weights[i] = 2.0*rand()/RAND_MAX-1;
            }
        }
    }
}