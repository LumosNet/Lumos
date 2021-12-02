#include "connect_layer.h"

void forward_connect_layer(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        l->output[i] = gemm(l->kernel_weights, l->input[i]);
        if (l->bias){
            add_bias(l->output[i], l->bias_weights, l->ksize, 1);
        }
        activate_tensor(l->output[i], l->active);
    }
}

void backward_connect_layer(Layer *l, Network *net)
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

Layer *make_connect_layer(LayerParams *p, int h, int w, int c)
{
    Layer *layer = NULL;
    if (0 == strcmp(p->type, "connect")){
        Layer *l = malloc(sizeof(Layer));
        l->type = CONNECT;
        l->input_h = h;
        l->input_w = w;
        l->input_c = c;
        Node *n = p->head;
        while (n){
            Params *param = n->val;
            if (0 == strcmp(param->key, "output")){
                l->ksize = atoi(param->val);
            } else if (0 == strcmp(param->key, "active")){
                Activation type = load_activate_type(param->val);
                l->active = load_activate(type);
                l->gradient = load_gradient(type);
            }
            n = n->next;
        }
        l->output_h = 1;
        l->output_w = l->ksize;
        l->output_c = 1;

        l->forward = forward_connect_layer;
        l->backward = backward_connect_layer;
        layer = l;
    }
    fprintf(stderr, "  connect             %2d      %4d x%4d         ->  %4d x%4d\n", \
            layer->ksize, 1, h*w*c, layer->output_h, layer->output_w);
    return layer;
}

void save_connect_weights(Layer *l, FILE *file)
{
    Tensor *kernel_weights = l->kernel_weights;
    Tensor *bias_weights = l->bias_weights;
    fwrite(kernel_weights->data, sizeof(float), kernel_weights->num, file);
    fwrite(bias_weights, sizeof(float), bias_weights->num, file);
}

void load_connect_weights(Layer *l, FILE *file)
{
    int size_k[] = {l->input_w, l->output_w};
    int size_b[] = {1, l->output_w};
    Tensor *kernel_weights = tensor_x(2, size_k, 0);
    Tensor *bias_weights = tensor_x(2, size_b, 0);
    fread(kernel_weights->data, sizeof(float), kernel_weights->num, file);
    fread(bias_weights->data, sizeof(float), bias_weights->num, file);
    l->kernel_weights = kernel_weights;
    l->bias_weights = bias_weights;
}