#include "convolutional_layer.h"

Layer *make_convolutional_layer(int filters, int ksize, int stride, int pad, int bias, char *active)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = CONVOLUTIONAL;
    l->filters = filters;
    l->ksize = ksize;
    l->stride = stride;
    l->pad = pad;
    l->bias = bias;

    Activation type = load_activate_type(active);
    l->active = type;

    l->initialize = init_convolutional_layer;
    l->forward = forward_convolutional_layer;
    l->backward = backward_convolutional_layer;

    l->initializegpu = init_convolutional_layer_gpu;
    l->forwardgpu = forward_convolutional_layer_gpu;
    l->backwardgpu = backward_convolutional_layer_gpu;

    l->weightinit = weightinit_convolutional_layer;
    l->weightinitgpu = weightinit_convolutional_layer_gpu;

    l->update = update_convolutional_layer_weights;
    l->updategpu = update_convolutional_layer_weights_gpu;

    l->saveweights = save_convolutional_layer_weights;
    l->saveweightsgpu = save_convolutional_layer_weights_gpu;

    fprintf(stderr, "Convolutional   Layer    :    [filters=%2d, ksize=%2d, stride=%2d, pad=%2d, bias=%d, active=%s]\n",
            l->filters, l->ksize, l->stride, l->pad, l->bias, active);
    return l;
}

void init_convolutional_layer(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = (l->input_h + 2 * l->pad - l->ksize) / l->stride + 1;
    l->output_w = (l->input_w + 2 * l->pad - l->ksize) / l->stride + 1;
    l->output_c = l->filters;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = l->ksize*l->ksize*l->input_c*l->output_h*l->output_w + l->filters*l->ksize*l->ksize*l->input_c;

    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));
    l->kernel_weights = calloc(l->filters*l->ksize*l->ksize*l->input_c, sizeof(float));
    l->update_kernel_weights = calloc(l->filters*l->ksize*l->ksize*l->input_c, sizeof(float));
    if (l->bias){
        l->bias_weights = calloc(l->filters, sizeof(float));
        l->update_bias_weights = calloc(l->filters, sizeof(float));
    }

    fprintf(stderr, "Convolutional   Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void weightinit_convolutional_layer(Layer l, FILE *fp)
{
    if (fp){
        fread(l.kernel_weights, sizeof(float), l.filters*l.ksize*l.ksize*l.input_c, fp);
        memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
        if (l.bias){
            fread(l.bias_weights, sizeof(float), l.filters, fp);
            memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
        }
        return;
    }
    float scale = sqrt((float)2 / (l.ksize*l.ksize*l.input_c));
    for (int i = 0; i < l.filters; ++i){
        float *weight = l.kernel_weights + i*l.input_c*l.ksize*l.ksize;
        for (int j = 0; j < l.ksize*l.ksize; ++j){
            weight[j] = scale*rand_normal();
        }
        for (int j = 0; j < l.input_c-1; ++j){
            float *weight_c = weight + (j+1)*l.ksize*l.ksize;
            memcpy(weight_c, weight, l.ksize*l.ksize*sizeof(float));
        }
    }
    if (l.bias){
        fill_cpu(l.bias_weights, l.filters, 0.001, 1);
        memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
}

void forward_convolutional_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        im2col(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
        gemm(0, 0, l.filters, l.ksize * l.ksize * l.input_c, l.ksize * l.ksize * l.input_c, l.output_h * l.output_w, 1,
             l.kernel_weights, l.workspace, output);
        if (l.bias){
            add_bias(output, l.bias_weights, l.filters, l.output_h * l.output_w);
        }
        activate_list(output, l.outputs, l.active);
    }
}

void backward_convolutional_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        gradient_list(output, l.outputs, l.active);
        matrix_multiply_cpu(delta_n, output, l.outputs, delta_n);
        gemm(1, 0, l.filters, l.ksize * l.ksize * l.input_c,
             l.filters, l.output_h * l.output_w, 1,
             l.kernel_weights, delta_n, l.workspace);
        col2im(l.workspace, l.ksize, l.stride, l.pad, l.input_h, l.input_w, l.input_c, delta_l);
    }
    update_convolutional_layer(l, rate, num, n_delta);
}

void update_convolutional_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_n = n_delta + offset_o;
        im2col(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
        gemm(0, 1, l.filters, l.output_h * l.output_w,
             l.ksize * l.ksize * l.input_c, l.output_h * l.output_w, 1,
             delta_n, l.workspace, l.workspace + l.ksize * l.ksize * l.input_c * l.output_h * l.output_w);
        saxpy_cpu(l.update_kernel_weights, l.workspace + l.ksize * l.ksize * l.input_c * l.output_h * l.output_w, l.filters * l.ksize * l.ksize * l.input_c, rate, l.update_kernel_weights);
        if (l.bias){
            sum_channel_cpu(delta_n, l.output_h, l.output_w, l.output_c, rate, l.workspace);
            add_bias(l.update_bias_weights, l.workspace, l.output_c, 1);
        }
    }
}

void update_convolutional_layer_weights(Layer l)
{
    memcpy(l.kernel_weights, l.update_kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
    if (l.bias){
        memcpy(l.bias_weights, l.update_bias_weights, l.filters*sizeof(float));
    }
}

void save_convolutional_layer_weights(Layer l, FILE *fp)
{
    fwrite(l.kernel_weights, sizeof(float), l.ksize*l.ksize*l.filters*l.input_c, fp);
    if (l.bias){
        fwrite(l.bias_weights, sizeof(float), l.filters, fp);
    }
}
