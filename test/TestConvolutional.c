#include <stdio.h>
#include <stdlib.h>

#include "lumos.h"
#include "network.h"
#include "convolutional_layer.h"
#include "active.h"

/*
    观察output是否正确
*/
void TestForwardConvolutionalLayer()
{
    int batch = 1;
    Layer l = {0};
    Network net = {0};
    l.type = CONVOLUTIONAL;
    l.input_h = 4;
    l.input_w = 4;
    l.input_c = 1;

    l.filters = 1;
    l.ksize = 3;
    l.stride = 1;
    l.pad = 0;
    l.bias = 1;
    l.batchnorm = 1;
    l.active = relu_activate;
    l.gradient = relu_gradient;

    l.output_h = (l.input_h + 2*l.pad - l.ksize) / l.stride + 1;
    l.output_w = (l.input_w + 2*l.pad - l.ksize) / l.stride + 1;
    l.output_c = l.filters;

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.lweights = load_convolutional_weights;
    l.sweights = save_convolutional_weights;
    l.update = update_convolutional_layer;

    l.workspace_size = l.ksize*l.ksize*l.input_c*l.output_h*l.output_w + l.filters*l.ksize*l.ksize*l.input_c;
    l.inputs = l.input_c*l.input_h*l.input_w;
    l.outputs = l.output_c*l.output_h*l.output_w;

    int size_k = l.filters*l.ksize*l.ksize*l.input_c;
    l.kernel_weights = calloc(size_k, sizeof(float));
    l.bias_weights = calloc(l.filters, sizeof(float));

    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.inputs, sizeof(float));

    fprintf(stderr, "  conv  %5d     %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", \
            l.filters, l.ksize, l.ksize, l.stride, l.input_h, \
            l.input_w, l.input_c, l.output_h, l.output_w, l.output_c);

    net.batch = batch;
    net.workspace = calloc(l.workspace_size, sizeof(float));
    net.input = calloc(l.input_h*l.input_w*l.input_c, sizeof(float));
    for (int i = 0; i < l.input_h*l.input_w*l.input_c; ++i){
        net.input[i] = i+1;
    }

    for (int i = 0; i < size_k; ++i){
        l.kernel_weights[i] = i+1;
    }
    l.bias_weights[0] = 1;
    l.input = net.input;
    l.forward(l, net);
}

/*
    观察权重更新是否正确
    观察l.delta是否正确
*/
void TestBackwardConvolutionalLayer()
{
    int batch = 1;
    Layer l = {0};
    Network net = {0};
    l.type = CONVOLUTIONAL;
    l.input_h = 4;
    l.input_w = 4;
    l.input_c = 1;

    l.filters = 1;
    l.ksize = 3;
    l.stride = 1;
    l.pad = 0;
    l.bias = 1;
    l.batchnorm = 1;
    l.active = relu_activate;
    l.gradient = relu_gradient;

    l.output_h = (l.input_h + 2*l.pad - l.ksize) / l.stride + 1;
    l.output_w = (l.input_w + 2*l.pad - l.ksize) / l.stride + 1;
    l.output_c = l.filters;

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.lweights = load_convolutional_weights;
    l.sweights = save_convolutional_weights;
    l.update = update_convolutional_layer;

    l.workspace_size = l.ksize*l.ksize*l.input_c*l.output_h*l.output_w + l.filters*l.ksize*l.ksize*l.input_c;
    l.inputs = l.input_c*l.input_h*l.input_w;
    l.outputs = l.output_c*l.output_h*l.output_w;

    int size_k = l.filters*l.ksize*l.ksize*l.input_c;
    l.kernel_weights = calloc(size_k, sizeof(float));
    l.bias_weights = calloc(l.filters, sizeof(float));

    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.inputs, sizeof(float));

    fprintf(stderr, "  conv  %5d     %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", \
            l.filters, l.ksize, l.ksize, l.stride, l.input_h, \
            l.input_w, l.input_c, l.output_h, l.output_w, l.output_c);

    net.batch = batch;
    net.learning_rate = 0.01;
    net.workspace_size = l.workspace_size;
    net.workspace = calloc(l.workspace_size, sizeof(float));
    net.delta = calloc(batch*l.outputs, sizeof(float));
    for (int i = 0; i < batch*l.outputs; ++i){
        net.delta[i] = i+1;
    }

    for (int i = 0; i < batch*l.outputs; ++i){
        l.output[i] = i+1;
    }
    for (int i = 0; i < size_k; ++i){
        l.kernel_weights[i] = i+1;
    }
    l.bias_weights[0] = 1;
    l.backward(l, net);
}
