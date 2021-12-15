#include "convolutional_layer.h"

void forward_convolutional_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        im2col(l.input[i]->data, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, net.workspace);
        gemm(0, 0, l.filters, l.ksize*l.ksize*l.input_c, l.ksize*l.ksize*l.input_c, l.output_h*l.output_w, 1, 
            l.kernel_weights->data, net.workspace, l.output[i]->data);
        if (l.bias){
            add_bias(l.output[i]->data, l.bias_weights->data, l.filters, l.output_h*l.output_w);
        }
        activate_list(l.output[i]->data, l.output[i]->num, l.active);
    }
}

void backward_convolutional_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        gradient_list(l.output[i]->data, l.output[i]->num, l.gradient);
        multiply(net.delta[i]->data, l.output[i]->data, l.output[i]->num, net.delta[i]->data);
        gemm(1, 0, l.kernel_weights->size[1], l.kernel_weights->size[0], 
            net.delta[i]->size[1], net.delta[i]->size[0], 1, 
            l.kernel_weights->data, net.delta[i]->data, net.workspace);
        col2im(net.workspace, l.ksize, l.stride, l.pad, l.input_h, l.input_w, l.input_c, l.delta[i]->data);
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

void update_convolutional_layer(Layer l, Network net)
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

// void save_convolutional_weights(Layer *l, FILE *file)
// {
//     for (int i = 0; i < l->filters; ++i){
//         for (int j = 0; j < l->ksize*l->ksize; ++j){
//             Tensor *kernel_weights = l->kernel_weights;
//             int index[] = {j+1, i+1};
//             int offset = index_ts2ls(index, 2, kernel_weights->size);
//             fread(kernel_weights+offset, sizeof(float), 1, file);
//         }
//     }
//     Tensor *bias_weights = l->bias_weights;
//     fread(bias_weights, sizeof(float), bias_weights->num, file);
// }

// void load_convolutional_weights(Layer *l, FILE *file)
// {
//     int size_k[] = {l->ksize*l->ksize*l->input_c, l->filters};
//     int size_b[] = {1, l->filters};
//     Tensor *kernel_weights = tensor_x(2, size_k, 0);

//     int weights_num = l->ksize*l->ksize*l->filters + l->filters;
//     float *weights = malloc(weights_num*sizeof(float));
//     fread(weights, sizeof(float), weights_num, file);

//     for (int i = 0; i < size_k[0]; ++i){
//         for (int j = 0; j < size_k[1]; ++j){
//             int kernels_offset = j*l->ksize*l->ksize;
//             int kerneli_offset = i % (l->ksize*l->ksize);
//             ts_change_pixel_ar(kernel_weights, i+1, j+1, weights[kernels_offset+kerneli_offset]);
//         }
//     }

//     Tensor *bias_weights = tensor_list(2, size_b, weights+l->ksize*l->ksize*l->filters);
//     l->kernel_weights = kernel_weights;
//     l->bias_weights = bias_weights;
// }