#include "convolutional_layer.h"

void forward_convolutional_layer(Layer l, Network net)
{
    printf("convolutional\n");
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        im2col(l.input+offset_i, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, net.workspace);
        gemm(0, 0, l.filters, l.ksize*l.ksize*l.input_c, l.ksize*l.ksize*l.input_c, l.output_h*l.output_w, 1, 
            l.kernel_weights, net.workspace, l.output+offset_o);
        if (l.bias){
            add_bias(l.output+offset_o, l.bias_weights, l.filters, l.output_h*l.output_w);
        }
        activate_list(l.output+offset_o, l.output_h*l.output_w*l.output_c, l.active);
    }
}

void backward_convolutional_layer(Layer l, Network net)
{
    printf("batc\n");
    for (int i = 0; i < net.batch; ++i){
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        int offset_d = i*l.input_h*l.input_w*l.input_c;
        gradient_list(l.output+offset_o, l.output_h*l.output_w*l.output_c, l.gradient);
        multiply(net.delta+offset_o, l.output+offset_o, l.output_h*l.output_w*l.output_c, net.delta+offset_o);
        gemm(1, 0, l.ksize*l.ksize*l.input_c, l.filters, 
            l.output_h, l.output_w, 1, 
            l.kernel_weights, net.delta+offset_o, net.workspace);
        col2im(net.workspace, l.ksize, l.stride, l.pad, l.input_h, l.input_w, l.input_c, l.delta+offset_d);
    }
    l.update(l, net);
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
    l.update = update_convolutional_layer;

    l.workspace_size = l.ksize*l.ksize*l.input_c*l.output_h*l.output_w;

    int size_k = l.filters*l.ksize*l.ksize*l.input_c;
    int size_b = l.filters;
    l.kernel_weights = calloc(size_k, sizeof(float));
    l.bias_weights = calloc(size_b, sizeof(float));

    int size_o = l.output_w * l.output_h * l.output_c;
    int size_d = l.input_w * l.input_h * l.input_c;
    l.output = calloc(batch*size_o, sizeof(float));
    l.delta = calloc(batch*size_d, sizeof(float));

    fprintf(stderr, "  conv  %5d     %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", \
            l.filters, l.ksize, l.ksize, l.stride, l.input_h, \
            l.input_w, l.input_c, l.output_h, l.output_w, l.output_c);
    return l;
}

void update_convolutional_layer(Layer l, Network net)
{
    float rate = net.learning_rate / (float)net.batch;
    for (int i = 0; i < net.batch; ++i){
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        gemm(0, 0, l.output_h, l.output_w, \
            l.input_h, l.input_w, 1, \
            net.delta+offset_o, l.input+offset_i, net.workspace);
        saxpy(l.kernel_weights, net.workspace, l.filters*l.ksize*l.ksize*l.input_c, rate, l.kernel_weights);
        if (l.bias){
            saxpy(l.bias_weights, net.delta+offset_o, l.filters, rate, l.bias_weights);
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
    fwrite(l.bias_weights, sizeof(float), l.filters, file);
}

void load_convolutional_weights(Layer l, FILE *file)
{
    if (file){
        float *weights = malloc(l.ksize*l.ksize*l.filters*sizeof(float));
        fread(weights, sizeof(float), l.ksize*l.ksize*l.filters, file);
        for (int i = 0; i < l.filters; ++i){
            for (int j = 0; j < l.input_c; ++j){
                for (int k = 0; k < l.ksize*l.ksize; ++k){
                    l.kernel_weights[i*l.ksize*l.ksize*l.input_c + j*l.ksize*l.ksize + k] = weights[i*l.ksize*l.ksize + k];
                }
            }
        }
        fread(l.bias_weights, sizeof(float), l.filters, file);
        free(weights);
    } else{
        float *weights = malloc(l.ksize*l.ksize*l.filters*sizeof(float));
        for (int i = 0; i < l.ksize*l.ksize*l.filters; ++i){
            weights[i] = rand()*0.01;
        }
        for (int i = 0; i < l.filters; ++i){
            for (int j = 0; j < l.input_c; ++j){
                for (int k = 0; k < l.ksize*l.ksize; ++k){
                    l.kernel_weights[i*l.ksize*l.ksize*l.input_c + j*l.ksize*l.ksize + k] = weights[i*l.ksize*l.ksize + k];
                }
            }
        }
        for (int i = 0; i < l.filters; ++i){
            l.bias_weights[i] = rand()*0.01;
        }
    }
}