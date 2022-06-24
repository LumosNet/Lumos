#include "convolutional_layer.h"

Layer *make_convolutional_layer(int filters, int ksize, int stride, int pad, int bias, int normalization, char *active)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = CONVOLUTIONAL;
    l->weights = 1;

    l->filters = filters;
    l->ksize = ksize;
    l->stride = stride;
    l->pad = pad;

    l->bias = bias;

    l->batchnorm = normalization;

    l->active_str = active;
    Activation type = load_activate_type(active);
    l->active = load_activate(type);
    l->gradient = load_gradient(type);

    l->forward = forward_convolutional_layer;
    l->backward = backward_convolutional_layer;
    l->update = update_convolutional_layer;

    l->init_layer_weights = init_convolutional_weights;

    restore_convolutional_layer(l);

    fprintf(stderr, "Convolutional   Layer    :    [filters=%2d, ksize=%2d, stride=%2d, pad=%2d, bias=%d, normalization=%d, active=%s]\n", \
            l->filters, l->ksize, l->stride, l->pad, l->bias, l->batchnorm, l->active_str);
    return l;
}

Layer *make_convolutional_layer_by_cfg(CFGParams *p)
{
    int filters = 0;
    int ksize = 0;
    int stride = 0;
    int pad = 0;

    int bias = 0;

    int normalization = 0;

    char *active = NULL;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "filters")){
            filters = atoi(param->val);
        } else if (0 == strcmp(param->key, "ksize")){
            ksize = atoi(param->val);
        } else if (0 == strcmp(param->key, "stride")){
            stride = atoi(param->val);
        } else if (0 == strcmp(param->key, "pad")){
            pad = atoi(param->val);
        } else if (0 == strcmp(param->key, "bias")){
            bias = atoi(param->val);
        } else if (0 == strcmp(param->key, "normalization")){
            normalization = atoi(param->val);
        } else if (0 == strcmp(param->key, "active")){
            active = param->val;
        }
        param = param->next;
    }

    Layer *l = make_convolutional_layer(filters, ksize, stride, pad, bias, normalization, active);
    return l;
}

void init_convolutional_layer(Layer *l, int w, int h, int c)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h*l->input_w*l->input_c;

    l->output_h = (l->input_h + 2*l->pad - l->ksize) / l->stride + 1;
    l->output_w = (l->input_w + 2*l->pad - l->ksize) / l->stride + 1;
    l->output_c = l->filters;
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->workspace_size = l->ksize*l->ksize*l->input_c*l->output_h*l->output_w + l->filters*l->ksize*l->ksize*l->input_c;

    l->kernel_weights_size = l->ksize*l->ksize*l->input_c;
    l->bias_weights_size = l->filters;
    l->deltas = l->inputs;

    fprintf(stderr, "Convolutional   Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void restore_convolutional_layer(Layer *l)
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

void init_convolutional_weights(Layer *l)
{
    random(1, l->inputs, 0.01, l->kernel_weights_size, l->kernel_weights);
    for (int i = 0; i < l->bias_weights_size; ++i){
        l->bias_weights[i] = 0.01;
    }
}

void forward_convolutional_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int input_offset = i*l.inputs;
        int output_offset = i*l.outputs;
        float *input = l.input+input_offset;
        float *output = l.output+output_offset;
        im2col(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
        gemm(0, 0, l.filters, l.ksize*l.ksize*l.input_c, l.ksize*l.ksize*l.input_c, l.output_h*l.output_w, 1, 
            l.kernel_weights, l.workspace, output);
        if (l.bias){
            add_bias(output, l.bias_weights, l.filters, l.output_h*l.output_w);
        }
        activate_list(output, l.outputs, l.active);
    }
}

void backward_convolutional_layer(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        gradient_list(l.output+offset_o, l.outputs, l.gradient);
        multiply(n_delta+offset_o, l.output+offset_o, l.outputs, n_delta+offset_o);
        gemm(1, 0, l.filters, l.ksize*l.ksize*l.input_c, 
            l.filters, l.output_h*l.output_w, 1, 
            l.kernel_weights, n_delta+offset_o, l.workspace);
        col2im(l.workspace, l.ksize, l.stride, l.pad, l.input_h, l.input_w, l.input_c, l.delta+offset_i);
    }
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
