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
    float rate = net->learning_rate / (float)net->batch;
    for (int i = 0; i < net->batch; ++i){
        gradient_tensor(l->output[i], l->gradient);
        Tensor *delta = net->delta[i];
        Tensor *d = gemm(delta, l->output[i]);
        Tensor *k_weights = tensor_copy(l->kernel_weights);
        Tensor *input = tensor_copy(l->input[i]);
        transposition(input);
        transposition(k_weights);
        net->delta[i] = gemm(d, k_weights);
        Tensor *d_w = gemm(d, input);
        saxpy(l->kernel_weights, d_w, rate);
        saxpy(l->bias_weights, d, rate);
        free_tensor(delta);
        free_tensor(d);
        free_tensor(k_weights);
        free_tensor(input);
        free_tensor(d_w);
    }
}

Layer make_connect_layer(LayerParams *p, int batch, int h, int w, int c)
{
    Layer l = {0};
    l.type = CONNECT;
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;
    Node *n = p->head;
    while (n){
        Params *param = n->val;
        if (0 == strcmp(param->key, "output")){
            l.ksize = atoi(param->val);
        } else if (0 == strcmp(param->key, "active")){
            Activation type = load_activate_type(param->val);
            l.active = load_activate(type);
            l.gradient = load_gradient(type);
        }
        n = n->next;
    }
    l.output_h = l.ksize;
    l.output_w = 1;
    l.output_c = 1;

    l.forward = forward_connect_layer;
    l.backward = backward_connect_layer;

    int size_o[] = {l.output_w, l.output_h, l.output_c};
    int size_d[] = {l.input_w, l.input_h, l.input_c};
    l.output = malloc(batch*sizeof(struct Tensor *));
    l.delta = malloc(batch*sizeof(struct Tensor *));
    for (int i = 0; i < batch; ++i){
        l.output[i] = tensor_x(3, size_o, 0);
        l.delta[i] = tensor_x(3, size_d, 0);
    }

    fprintf(stderr, "  connect             %2d      %4d x%4d         ->  %4d x%4d\n", \
            l.ksize, 1, h*w*c, l.output_h, l.output_w);
    return l;
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