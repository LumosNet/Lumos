#include "convolutional_layer.h"

Layer make_convolutional_layer(CFGParams *p)
{
    Layer l = {0};
    l.type = CONVOLUTIONAL;
    l.weights = 1;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "filters")){
            l.filters = atoi(param->val);
        } else if (0 == strcmp(param->key, "ksize")){
            l.ksize = atoi(param->val);
        } else if (0 == strcmp(param->key, "stride")){
            l.stride = atoi(param->val);
        } else if (0 == strcmp(param->key, "pad")){
            l.pad = atoi(param->val);
        } else if (0 == strcmp(param->key, "bias")){
            l.bias = atoi(param->val);
        } else if (0 == strcmp(param->key, "normalization")){
            l.batchnorm = atoi(param->val);
        } else if (0 == strcmp(param->key, "active")){
            l.active_str = param->key;
            Activation type = load_activate_type(param->val);
            l.active = load_activate(type);
            l.gradient = load_gradient(type);
        }
        param = param->next;
    }

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;

    restore_convolutional_layer(l);

    fprintf(stderr, "Convolutional Layer: [filters=%d, ksize=%d, stride=%d, pad=%d, bias=%d, normalization=%d, active=%s]\n", \
            l.filters, l.ksize, l.stride, l.pad, l.bias, l.batchnorm, l.active);
    return l;
}

void init_convolutional_layer(Layer l, int w, int h, int c)
{
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;
    l.inputs = l.input_h*l.input_w*l.input_c;

    l.output_h = (l.input_h + 2*l.pad - l.ksize) / l.stride + 1;
    l.output_w = (l.input_w + 2*l.pad - l.ksize) / l.stride + 1;
    l.output_c = l.filters;
    l.outputs = l.output_h*l.output_w*l.output_c;

    l.workspace_size = l.ksize*l.ksize*l.input_c*l.output_h*l.output_w + l.filters*l.ksize*l.ksize*l.input_c;

    l.kernel_weights_size = l.ksize*l.ksize*l.input_c;
    l.bias_weights_size = l.filters;
    l.deltas = l.inputs;
}

void restore_convolutional_layer(Layer l)
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

    l.kernel_weights_size = -1;
    l.bias_weights_size = -1;
    l.deltas = -1;

    l.input = NULL;
    l.output = NULL;
    l.kernel_weights = NULL;
    l.bias_weights = NULL;
    l.delta = NULL;
}

void forward_convolutional_layer(Layer l)
{
    im2col(l.input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
    gemm(0, 0, l.filters, l.ksize*l.ksize*l.input_c, l.ksize*l.ksize*l.input_c, l.output_h*l.output_w, 1, 
        l.kernel_weights, l.workspace, l.output);
    if (l.bias){
        add_bias(l.output, l.bias_weights, l.filters, l.output_h*l.output_w);
    }
    activate_list(l.output, l.outputs, l.active);
}

void backward_convolutional_layer(Layer l, float *n_delta)
{
    gradient_list(l.output, l.outputs, l.gradient);
    multiply(n_delta, l.output, l.outputs, n_delta);
    gemm(1, 0, l.filters, l.ksize*l.ksize*l.input_c, 
        l.filters, l.output_h*l.output_w, 1, 
        l.kernel_weights, n_delta, l.workspace);
    col2im(l.workspace, l.ksize, l.stride, l.pad, l.input_h, l.input_w, l.input_c, l.delta);
}

void update_convolutional_layer(Layer l, float rate, float *n_delta)
{
    im2col(l.input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
    gemm(0, 1, l.filters, l.output_h*l.output_w, \
        l.ksize*l.ksize*l.input_c, l.output_h*l.output_w, 1, \
        n_delta, l.workspace, l.workspace+l.ksize*l.ksize*l.input_c*l.output_h*l.output_w);
    saxpy(l.update_kernel_weights, l.workspace+l.ksize*l.ksize*l.input_c*l.output_h*l.output_w, l.filters*l.ksize*l.ksize*l.input_c, rate, l.update_kernel_weights);
    if (l.bias){
        for (int j = 0; j < l.filters; ++j){
            float bias = sum_cpu(n_delta+j*l.output_h*l.output_w, l.output_h*l.output_w);
            l.update_bias_weights[j] += bias * rate;
        }
    }
}
