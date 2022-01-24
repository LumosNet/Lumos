#include "connect_layer.h"

void forward_connect_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        gemm(0, 0, l.output_h, l.input_h, l.input_h, l.input_w, 
            1, l.kernel_weights, l.input+offset_i, l.output+offset_o);
        if (l.bias){
            add_bias(l.output+offset_o, l.bias_weights, l.ksize, 1);
        }
        activate_list(l.output+offset_o, l.output_w*l.output_h*l.output_c, l.active);
    }
}

void backward_connect_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        int offset_d = i*l.input_h*l.input_w*l.input_c;
        gradient_list(l.output+offset_o, l.output_h*l.output_w*l.output_c, l.gradient);
        multiply(net.delta+offset_o, l.output+offset_o, l.output_h*l.output_w*l.output_c, net.delta+offset_o);
        gemm(1, 0, l.output_h, l.input_h, l.output_h, l.input_w, 1, 
            l.kernel_weights, net.delta+offset_o, l.delta+offset_d);
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
    l.filters = 1;
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
    l.lweights = load_connect_weights;
    l.sweights = save_connect_weights;
    l.update = update_connect_layer;

    l.workspace_size = l.input_c*l.input_h*l.input_w*l.output_c*l.output_h*l.output_w;

    int size_k = l.input_h * l.output_h;
    int size_b = l.output_h;
    l.kernel_weights = calloc(size_k, sizeof(float));
    l.bias_weights = calloc(size_b, sizeof(float));

    int size_o = l.output_w * l.output_h * l.output_c;
    int size_d = l.input_w * l.input_h * l.input_c;
    l.output = calloc(batch*size_o, sizeof(float));
    l.delta = calloc(batch*size_d, sizeof(float));

    fprintf(stderr, "  connect             %2d      %4d x%4d         ->  %4d x%4d\n", \
            l.ksize, 1, h*w*c, l.output_w, l.output_h);
    return l;
}

void update_connect_layer(Layer l, Network net)
{
    float rate = -net.learning_rate / (float)net.batch;
    for (int i = 0; i < net.batch; ++i){
        int offset_d = i*l.output_h*l.output_w*l.output_c;
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        gemm(0, 0, l.output_h, l.output_w, \
            l.input_h, l.input_w, 1, \
            net.delta+offset_d, l.input+offset_i, net.workspace);
        saxpy(l.kernel_weights, net.workspace, l.output_h * l.input_h, rate, l.kernel_weights);
        if (l.bias){
            saxpy(l.bias_weights, net.delta+offset_d, l.output_h, rate, l.bias_weights);
        }
    }
}

void save_connect_weights(Layer l, FILE *file)
{
    fwrite(l.kernel_weights, sizeof(float), l.output_h * l.input_h, file);
    fwrite(l.bias_weights, sizeof(float), l.output_h, file);
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
        for (int i = 0; i < l.output_h; ++i){
            l.bias_weights[i] = 2.0*rand()/RAND_MAX-1;
        }
    }
}