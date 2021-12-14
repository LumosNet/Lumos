#include "convolutional_layer.h"

void forward_convolutional_layer(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *output = l->output[i];
        Tensor *kernel = l->kernel_weights;
        Tensor *colimg = l->colimg;
        im2col(l->input[i], kernel->size[0], l->stride, l->pad, colimg->data);
        gemm(colimg, kernel, output->data);
        if (l->bias){
            add_bias(output, l->bias_weights, l->filters, l->output_h*l->output_w);
        }
        activate_tensor(output, l->active);
        transposition(output);
        int size[] = {l->output_w, l->output_h, l->output_c};
        resize_ts(output, 3, size);
    }
}

void backward_convolutional_layer(Layer *l, Network *net)
{
    float rate = net->learning_rate / (float)net->batch;
    Tensor *delta_i = l->delta_i;
    Tensor *derivative = l->derivative;
    Tensor *gradient = l->gradient_w;
    Tensor *k_weights = tensor_copy(l->kernel_weights);
    transposition(k_weights);
    for (int i = 0; i < net->batch; ++i){
        Tensor *output = l->output[i];
        Tensor *delta_k = net->delta[i];
        Tensor *delta = l->delta[i];
        gradient_tensor(output, l->gradient);
        gemm(delta, l->output[i], derivative->data);
        Tensor *input = tensor_copy(l->input[i]);
        transposition(input);
        gemm(derivative, k_weights, delta_i->data);
        col2im(delta_i, l->ksize, l->stride, l->pad, l->input_h, l->input_w, l->input_c, delta->data);
        gemm(derivative, input, gradient->data);
        ts_saxpy(l->kernel_weights, gradient, rate);
        ts_saxpy(l->bias_weights, derivative, rate);
        free_tensor(input);
    }
    free_tensor(k_weights);
}

Layer make_convolutional_layer(LayerParams *p, int batch, int h, int w, int c)
{
    Layer l = {0};
    l.type = CONVOLUTIONAL;
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;
    Node *n = p->head;
    while (n){
        Params *param = n->val;
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
        n = n->next;
    }
    l.output_h = (l.input_h + 2*l.pad - l.ksize) / l.stride + 1;
    l.output_w = (l.input_w + 2*l.pad - l.ksize) / l.stride + 1;
    l.output_c = l.filters;

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;

    int size_o[] = {l.output_w, l.output_h, l.output_c};
    int size_d[] = {l.input_w, l.input_h, l.input_c};
    l.output = malloc(batch*sizeof(Tensor *));
    l.delta = malloc(batch*sizeof(Tensor *));
    for (int i = 0; i < batch; ++i){
        l.output[i] = tensor_x(3, size_o, 0);
        l.delta[i] = tensor_x(3, size_d, 0);
    }
    fprintf(stderr, "  conv  %5d     %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", \
            l.filters, l.ksize, l.ksize, l.stride, l.input_h, \
            l.input_w, l.input_c, l.output_h, l.output_w, l.output_c);
    return l;
}

void save_convolutional_weights(Layer *l, FILE *file)
{
    for (int i = 0; i < l->filters; ++i){
        for (int j = 0; j < l->ksize*l->ksize; ++j){
            Tensor *kernel_weights = l->kernel_weights;
            int index[] = {j+1, i+1};
            int offset = index_ts2ls(index, 2, kernel_weights->size);
            fread(kernel_weights+offset, sizeof(float), 1, file);
        }
    }
    Tensor *bias_weights = l->bias_weights;
    fread(bias_weights, sizeof(float), bias_weights->num, file);
}

void load_convolutional_weights(Layer *l, FILE *file)
{
    int size_k[] = {l->ksize*l->ksize*l->input_c, l->filters};
    int size_b[] = {1, l->filters};
    Tensor *kernel_weights = tensor_x(2, size_k, 0);

    int weights_num = l->ksize*l->ksize*l->filters + l->filters;
    float *weights = malloc(weights_num*sizeof(float));
    fread(weights, sizeof(float), weights_num, file);

    for (int i = 0; i < size_k[0]; ++i){
        for (int j = 0; j < size_k[1]; ++j){
            int kernels_offset = j*l->ksize*l->ksize;
            int kerneli_offset = i % (l->ksize*l->ksize);
            ts_change_pixel_ar(kernel_weights, i+1, j+1, weights[kernels_offset+kerneli_offset]);
        }
    }

    Tensor *bias_weights = tensor_list(2, size_b, weights+l->ksize*l->ksize*l->filters);
    l->kernel_weights = kernel_weights;
    l->bias_weights = bias_weights;
}