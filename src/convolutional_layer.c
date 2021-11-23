#include "convolutional_layer.h"

void forward_convolutional_layer(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        l->output[i] = convolutional(l->input[i], l->kernel_weights, l->pad, l->stride);
        if (l->bias){
            add_bias(l->output[i], l->bias_weights, l->filters, l->output_h*l->output_w);
        }
        activate_tensor(l->output[i], l->active);
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
        net->delta[i] = gemm(d, k_weights);
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