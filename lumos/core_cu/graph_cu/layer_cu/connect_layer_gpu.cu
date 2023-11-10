#include "connect_layer_gpu.h"

Layer *make_connect_layer_gpu(int output, int bias, int normalize, char *active)
{
    Layer *l = (Layer*)malloc(sizeof(Layer));
    l->filters = 1;
    l->ksize = output;

    l->init = init_connect_layer_gpu;
    l->release = release_connect_layer_gpu;
    l->forward = forward_connect_layer_gpu;
    l->backward = backward_connect_layer_gpu;
    l->update = update_connect_layer_gpu;
    Activation type = load_activate_type(active);
    l->active = load_activate_gpu(type);
    l->gradient = load_gradient_gpu(type);

    l->bias = bias;
    l->batchnorm = normalize;

    fprintf(stderr, "Connect         Layer    :    [output=%4d, bias=%d, active=%s]\n", l->ksize, l->bias, active);
    return l;
}

void init_connect_layer_gpu(Layer *l, int w, int h, int c)
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

    cudaMalloc((void**)&l->output, l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, l->inputs*sizeof(float));
    cudaMalloc((void**)&l->kernel_weights, l->inputs*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->update_kernel_weights, l->inputs*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->bias_weights, l->outputs*sizeof(float));
    cudaMalloc((void**)&l->update_bias_weights, l->outputs*sizeof(float));

    fprintf(stderr, "Connect         Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void release_connect_layer_gpu(Layer *l)
{
    cudaFree(l->output);
    cudaFree(l->delta);
    cudaFree(l->kernel_weights);
    cudaFree(l->update_kernel_weights);
    cudaFree(l->bias_weights);
    cudaFree(l->update_bias_weights);
}

void forward_connect_layer_gpu(Layer l, int num)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        gemm_gpu(0, 0, l.outputs, l.inputs, l.inputs, 1,
             1, l.kernel_weights, input, output);
        if (l.bias){
            add_bias_gpu(output, l.bias_weights, l.ksize, 1);
        }
        activate_list_gpu(output, l.outputs, l.active);
    }
}

void backward_connect_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        gradient_list_gpu(output, l.outputs, l.gradient);
        matrix_multiply_gpu(delta_n, output, l.outputs, delta_n);
        gemm_gpu(1, 0, l.output_c, l.input_c, l.output_c, l.input_w, 1,
             l.kernel_weights, delta_n, delta_l);
    }
    l.update(l, rate, num, n_delta);
}

void update_connect_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_n = n_delta + offset_o;
        gemm_gpu(0, 1, l.output_c, l.output_w,
             l.input_c, l.input_w, 1,
             delta_n, input, l.workspace);
        saxpy_gpu(l.update_kernel_weights, l.workspace, l.output_c * l.input_c, rate, l.update_kernel_weights);
        if (l.bias){
            saxpy_gpu(l.update_bias_weights, delta_n, l.outputs, rate, l.update_bias_weights);
        }
    }
}
