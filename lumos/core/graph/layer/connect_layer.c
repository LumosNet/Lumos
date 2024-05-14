#include "connect_layer.h"

Layer *make_connect_layer(int output, int bias, char *active)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = CONNECT;
    l->ksize = output;
    l->bias = bias;

    Activation type = load_activate_type(active);
    l->active = type;

    l->initialize = init_connect_layer;
    l->forward = forward_connect_layer;
    l->backward = backward_connect_layer;

    l->initializegpu = init_connect_layer_gpu;
    l->forwardgpu = forward_connect_layer_gpu;
    l->backwardgpu = backward_connect_layer_gpu;

    l->weightinit = weightinit_connect_layer;
    l->weightinitgpu = weightinit_connect_layer_gpu;

    l->update = update_connect_layer_weights;
    l->updategpu = update_connect_layer_weights_gpu;

    l->saveweights = save_connect_layer_weights;
    l->saveweightsgpu = save_connect_layer_weights_gpu;

    fprintf(stderr, "Connect         Layer    :    [output=%4d, bias=%d, active=%s]\n", l->ksize, l->bias, active);
    return l;
}

void init_connect_layer(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = 1;
    l->output_w = 1;
    l->output_c = l->ksize;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = l->inputs * l->outputs;

    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));
    l->kernel_weights = calloc(l->inputs * l->outputs, sizeof(float));
    l->update_kernel_weights = calloc(l->inputs * l->outputs, sizeof(float));
    if (l->bias){
        l->bias_weights = calloc(l->outputs, sizeof(float));
        l->update_bias_weights = calloc(l->outputs, sizeof(float));
    }

    fprintf(stderr, "Connect         Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void weightinit_connect_layer(Layer l, FILE *fp)
{
    if (fp){
        fread(l.kernel_weights, sizeof(float), l.outputs*l.inputs, fp);
        memcpy(l.update_kernel_weights, l.kernel_weights, l.inputs*l.outputs*sizeof(float));
        if (l.bias){
            fread(l.bias_weights, sizeof(float), l.outputs, fp);
            memcpy(l.update_bias_weights, l.bias_weights, l.outputs*sizeof(float));
        }
        return;
    }
    float scale = sqrt((float)2 / l.inputs);
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        l.kernel_weights[i] = scale*rand_uniform(-1, 1);
    }
    if (l.bias){
        fill_cpu(l.bias_weights, l.outputs, 0.001, 1);
        memcpy(l.update_bias_weights, l.bias_weights, l.outputs*sizeof(float));
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.inputs*l.outputs*sizeof(float));
}

void forward_connect_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        gemm(0, 0, l.outputs, l.inputs, l.inputs, 1,
             1, l.kernel_weights, input, output);
        if (l.bias){
            add_bias(output, l.bias_weights, l.ksize, 1);
        }
        activate_list(output, l.outputs, l.active);
    }
}

void backward_connect_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        gradient_list(output, l.outputs, l.active);
        matrix_multiply_cpu(delta_n, output, l.outputs, delta_n);
        gemm(1, 0, l.output_c, l.input_c, l.output_c, l.input_w, 1,
             l.kernel_weights, delta_n, delta_l);
    }
    update_connect_layer(l, rate, num, n_delta);
}

void update_connect_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_n = n_delta + offset_o;
        gemm(0, 1, l.output_c, l.output_w,
             l.input_c, l.input_w, 1,
             delta_n, input, l.workspace);
        saxpy_cpu(l.update_kernel_weights, l.workspace, l.output_c * l.input_c, rate, l.update_kernel_weights);
        if (l.bias){
            saxpy_cpu(l.update_bias_weights, delta_n, l.outputs, rate, l.update_bias_weights);
        }
    }
}

void update_connect_layer_weights(Layer l)
{
    memcpy(l.kernel_weights, l.update_kernel_weights, l.inputs*l.outputs*sizeof(float));
    if (l.bias){
        memcpy(l.bias_weights, l.update_bias_weights, l.outputs*sizeof(float));
    }
}

void save_connect_layer_weights(Layer l, FILE *fp)
{
    fwrite(l.kernel_weights, sizeof(float), l.inputs*l.outputs, fp);
    if (l.bias){
        fwrite(l.bias_weights, sizeof(float), l.outputs, fp);
    }
}
