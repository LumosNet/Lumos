#include "connect_layer.h"

void forward_connect_layer(Layer l, Network net)
{
    printf("connect\n");
    for (int i = 0; i < net.batch; ++i){
        gemm(0, 0, l.kernel_weights->size[1], l.kernel_weights->size[0], 
            l.input[i]->size[1], l.input[i]->size[0], 
            1, l.kernel_weights->data, l.input[i]->data, l.output[i]->data);
        if (l.bias){
            add_bias(l.output[i]->data, l.bias_weights->data, l.ksize, 1);
        }
        activate_list(l.output[i]->data, l.output[i]->num, l.active);
    }
    printf("connect\n");
}

void backward_connect_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        gradient_list(l.output[i]->data, l.output[i]->num, l.gradient);
        multiply(net.delta[i]->data, l.output[i]->data, l.output[i]->num, net.delta[i]->data);
        gemm(1, 0, l.kernel_weights->size[1], l.kernel_weights->size[0], 
            net.delta[i]->size[1], net.delta[i]->size[0], 1, 
            l.kernel_weights->data, net.delta[i]->data, l.delta[i]->data);
    }
    l.update(l, net);
}

Layer make_connect_layer(LayerParams *p, int batch, int h, int w, int c)
{
    Layer l = {0};
    l.type = CONNECT;
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;
    l.bias = 1;
    Node *n = p->head;
    while (n){
        Params *param = n->val;
        if (0 == strcmp(param->key, "output")){
            l.ksize = atoi(param->val);
        } else if (0 == strcmp(param->key, "active")){
            Activation type = load_activate_type(param->val);
            l.active = load_activate(type);
            l.gradient = load_gradient(type);
        } else if (0 == strcmp(param->key, "bias")){
            l.bias = atoi(param->val);
        }
        n = n->next;
    }
    l.output_h = l.ksize;
    l.output_w = 1;
    l.output_c = 1;

    l.forward = forward_connect_layer;
    l.backward = backward_connect_layer;
    l.update = update_connect_layer;

    l.workspace_size = l.input_c*l.input_h*l.input_w*l.output_c*l.output_h*l.output_w;

    int size_k[] = {l.output_h, l.input_h};
    int size_b[] = {1, l.output_h};
    l.kernel_weights = tensor_x(2, size_k, 0);
    l.bias_weights = tensor_x(2, size_b, 0);

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

void update_connect_layer(Layer l, Network net)
{
    float rate = net.learning_rate / (float)net.batch;
    for (int i = 0; i < net.batch; ++i){
        gemm(0, 0, net.delta[i]->size[1], net.delta[i]->size[0], \
            l.input[i]->size[1], l.input[i]->size[0], 1, \
            net.delta[i]->data, l.input[i]->data, net.workspace);
        saxpy(l.kernel_weights->data, net.workspace, l.kernel_weights->num, rate, l.kernel_weights->data);
        if (l.bias){
            saxpy(l.bias_weights->data, net.delta[i]->data, l.bias_weights->num, rate, l.bias_weights->data);
        }
    }
}

void save_connect_weights(Layer l, FILE *file)
{
    fwrite(l.kernel_weights->data, sizeof(float), l.kernel_weights->num, file);
    fwrite(l.bias_weights->data, sizeof(float), l.bias_weights->num, file);
}

void load_connect_weights(Layer l, FILE *file)
{
    if (file){
        fread(l.kernel_weights->data, sizeof(float), l.kernel_weights->num, file);
        fread(l.bias_weights->data, sizeof(float), l.bias_weights->num, file);
    } else{
        for (int i = 0; i < l.output_h*l.input_h; ++i){
            l.kernel_weights->data[i] = rand()*0.001;
        }
        for (int i = 0; i < l.output_h; ++i){
            l.bias_weights->data[i] = rand()*0.001;
        }
    }
}