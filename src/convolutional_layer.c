#include "convolutional_layer.h"

void forward_convolutional_layer(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        l->output[i] = convolutional(l->input[i], l->kernel_weights, l->pad, l->stride);
        if (l->bias){
            add_bias(l->output[i], l->bias_weights, l->filters, l->output_h*l->output_w);
        }
        activate_tensor(l->output[i], l->active);
        transposition(l->output[i]);
        int size[] = {l->output_h, l->output_w, l->output_c};
        resize(l->input[i], 3, size);
    }
}

void backward_convolutional_layer(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        gradient_tensor(l->output[i], l->gradient);
        Tensor *delta = net->delta[i];
        Tensor *d = gemm(delta, l->output[i]);
        Tensor *k_weights = copy(l->kernel_weights);
        Tensor *input = copy(l->input[i]);
        transposition(input);
        transposition(k_weights);
        net->delta[i] = col2im(gemm(d, k_weights), l->ksize, l->stride, l->pad, l->input_h, l->input_w, l->input_c);
        Tensor *d_w = gemm(d, input);
        saxpy(l->kernel_weights, d_w, net->batch*net->learning_rate);
        saxpy(l->bias_weights, d, net->batch*net->learning_rate);
        del(delta);
        del(d);
        del(k_weights);
        del(input);
        del(d_w);
    }
}

Layer *make_convolutional_layer(LayerParams *p, int h, int w, int c)
{
    Layer *layer = NULL;
    if (0 == strcmp(p->type, "convolutional")){
        Layer *l = malloc(sizeof(Layer));
        l->type = CONVOLUTIONAL;
        l->input_h = h;
        l->input_w = w;
        l->input_c = c;
        Node *n = p->head;
        while (n){
            Params *param = n->val;
            if (0 == strcmp(param->key, "filters")){
                l->filters = atoi(param->val);
            } else if (0 == strcmp(param->key, "ksize")){
                l->ksize = atoi(param->val);
            } else if (0 == strcmp(param->key, "stride")){
                l->stride = atoi(param->val);
            } else if (0 == strcmp(param->key, "pad")){
                l->pad = atoi(param->val);
            } else if (0 == strcmp(param->key, "bias")){
                l->bias = atoi(param->val);
            } else if (0 == strcmp(param->key, "normalization")){
                l->batchnorm = atoi(param->val);
            } else if (0 == strcmp(param->key, "active")){
                Activation type = load_activate_type(param->val);
                l->active = load_activate(type);
                l->gradient = load_gradient(type);
            }
            n = n->next;
        }
        l->output_h = (l->input_h + 2*l->pad - l->ksize) / l->stride + 1;
        l->output_w = (l->input_w + 2*l->pad - l->ksize) / l->stride + 1;
        l->output_c = l->filters;
        l->forward = forward_convolutional_layer;
        l->backward = backward_convolutional_layer;
        layer = l;
    }
    fprintf(stderr, "  conv  %5d     %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", \
            layer->filters, layer->ksize, layer->ksize, layer->stride, layer->input_h, \
            layer->input_w, layer->input_c, layer->output_h, layer->output_w, layer->output_c);
    return layer;
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
            change_pixel_ar(kernel_weights, i+1, j+1, weights[kernels_offset+kerneli_offset]);
        }
    }

    Tensor *bias_weights = tensor_list(2, size_b, weights+l->ksize*l->ksize*l->filters);
    l->kernel_weights = kernel_weights;
    l->bias_weights = bias_weights;
}