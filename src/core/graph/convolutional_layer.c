#include "convolutional_layer.h"

Layer make_convolutional_layer(CFGParams *p, int h, int w, int c)
{
    Layer l = {0};
    l.type = CONVOLUTIONAL;
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;

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
            Activation type = load_activate_type(param->val);
            l.active = load_activate(type);
            l.gradient = load_gradient(type);
        }
        param = param->next;
    }
    l.output_h = (l.input_h + 2*l.pad - l.ksize) / l.stride + 1;
    l.output_w = (l.input_w + 2*l.pad - l.ksize) / l.stride + 1;
    l.output_c = l.filters;

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.lweights = load_convolutional_weights;
    l.sweights = save_convolutional_weights;
    l.update = update_convolutional_layer;

    l.workspace_size = l.ksize*l.ksize*l.input_c*l.output_h*l.output_w + l.filters*l.ksize*l.ksize*l.input_c;
    l.inputs = l.input_c*l.input_h*l.input_w;
    l.outputs = l.output_c*l.output_h*l.output_w;

    int size_k = l.filters*l.ksize*l.ksize*l.input_c;
    l.kernel_weights = calloc(size_k, sizeof(float));
    l.bias_weights = calloc(l.filters, sizeof(float));

    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.inputs, sizeof(float));

    fprintf(stderr, "  conv  %5d     %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", \
            l.filters, l.ksize, l.ksize, l.stride, l.input_h, \
            l.input_w, l.input_c, l.output_h, l.output_w, l.output_c);
    return l;
}

void forward_convolutional_layer(Layer l, float *workspace)
{
    im2col(l.input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, workspace);
    gemm(0, 0, l.filters, l.ksize*l.ksize*l.input_c, l.ksize*l.ksize*l.input_c, l.output_h*l.output_w, 1, 
        l.kernel_weights, workspace, l.output);
    if (l.bias){
        add_bias(l.output, l.bias_weights, l.filters, l.output_h*l.output_w);
    }
    activate_list(l.output, l.outputs, l.active);
}

void backward_convolutional_layer(Layer l, float *n_delta, float *workspace)
{
    gradient_list(l.output, l.outputs, l.gradient);
    multiply(net.delta, l.output, l.outputs, net.delta);
    gemm(1, 0, l.filters, l.ksize*l.ksize*l.input_c, 
        l.filters, l.output_h*l.output_w, 1, 
        l.kernel_weights, net.delta, workspace);
    col2im(workspace, l.ksize, l.stride, l.pad, l.input_h, l.input_w, l.input_c, l.delta);
}

void update_convolutional_layer(Layer l, float rate, float *n_delta, float *workspace)
{
    im2col(l.input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, net.workspace);
    gemm(0, 1, l.filters, l.output_h*l.output_w, \
        l.ksize*l.ksize*l.input_c, l.output_h*l.output_w, 1, \
        net.delta, net.workspace, net.workspace+l.ksize*l.ksize*l.input_c*l.output_h*l.output_w);
    saxpy(l.kernel_weights, net.workspace+l.ksize*l.ksize*l.input_c*l.output_h*l.output_w, l.filters*l.ksize*l.ksize*l.input_c, rate, l.kernel_weights);
    if (l.bias){
        for (int j = 0; j < l.filters; ++j){
            float bias = sum_cpu(net.delta+j*l.output_h*l.output_w, l.output_h*l.output_w);
            l.bias_weights[j] += bias * rate;
        }
    }
}

void save_convolutional_weights(Layer l, FILE *file)
{
    for (int i = 0; i < l.filters; ++i){
        for (int j = 0; j < l.ksize*l.ksize; ++j){
            int offset = i*l.ksize*l.ksize*l.input_c + j;
            fwrite(l.kernel_weights+offset, sizeof(float), 1, file);
        }
    }
    if (l.bias) fwrite(l.bias_weights, sizeof(float), l.filters, file);
}

void load_convolutional_weights(Layer l, FILE *file)
{
    float *weights = malloc(l.ksize*l.ksize*l.filters*sizeof(float));
    if (file){
        fread(weights, sizeof(float), l.ksize*l.ksize*l.filters, file);
        for (int i = 0; i < l.filters; ++i){
            for (int j = 0; j < l.input_c; ++j){
                for (int k = 0; k < l.ksize*l.ksize; ++k){
                    l.kernel_weights[i*l.ksize*l.ksize*l.input_c + j*l.ksize*l.ksize + k] = weights[i*l.ksize*l.ksize + k];
                }
            }
        }
        if (l.bias) fread(l.bias_weights, sizeof(float), l.filters, file);
    } else{
        for (int i = 0; i < l.ksize*l.ksize*l.filters; ++i){
            weights[i] = 2.0*rand()/RAND_MAX-1;
        }
        for (int i = 0; i < l.filters; ++i){
            for (int j = 0; j < l.input_c; ++j){
                for (int k = 0; k < l.ksize*l.ksize; ++k){
                    l.kernel_weights[i*l.ksize*l.ksize*l.input_c + j*l.ksize*l.ksize + k] = weights[i*l.ksize*l.ksize + k];
                }
            }
        }
        if (l.bias){
            for (int i = 0; i < l.filters; ++i){
                l.bias_weights[i] = 2.0*rand()/RAND_MAX-1;
            }
        }
    }
    free(weights);
}