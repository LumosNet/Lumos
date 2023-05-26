#include "connect_layer.h"

Layer *make_connect_layer(int output, int bias, int normalize, char *active)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = CONNECT;
    l->filters = 1;
    l->weights = 1;

    l->ksize = output;

    l->active_str = active;
    Activation type = load_activate_type(active);
    l->active = type;
    l->gradient = type;

    l->bias = bias;
    l->batchnorm = normalize;

    fprintf(stderr, "Connect         Layer    :    [output=%4d, bias=%d, active=%s]\n", l->ksize, l->bias, l->active_str);
    return l;
}

void init_connect_layer(Layer *l, int w, int h, int c)
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

    l->kernel_weights_size = l->inputs * l->outputs;
    l->normalize_weights_size = 0;
    if (l->batchnorm){
        l->bias = 1;
        l->normalize_weights_size = l->outputs;
    }
    l->bias_weights_size = 0;
    if (l->bias || l->batchnorm){
        l->bias_weights_size = l->outputs;
    }
    l->deltas = l->inputs;

    if (l->coretype == GPU){
        l->forward = forward_connect_layer_gpu;
        l->backward = backward_connect_layer_gpu;
        l->update = update_connect_layer_gpu;
    } else {
        l->forward = forward_connect_layer;
        l->backward = backward_connect_layer;
        l->update = update_connect_layer;
    }

    fprintf(stderr, "Connect         Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_connect_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        // for (int j = 0; j < l.kernel_weights_size; ++j)
        gemm(0, 0, l.outputs, l.inputs, l.inputs, 1,
             1, l.kernel_weights, input, output);
        if (l.batchnorm){
            forward_normalization_layer(l, num);
        }
        if (l.bias)
        {
            add_bias(output, l.bias_weights, l.ksize, 1);
        }
        activate_list(output, l.outputs, l.active);
    }
}

void backward_connect_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        gradient_list(output, l.outputs, l.gradient);
        if (l.batchnorm){
            backward_normalization_layer(l, rate, num, n_delta);
        }
        matrix_multiply_cpu(delta_n, output, l.outputs, delta_n);
        gemm(1, 0, l.output_c, l.input_c, l.output_c, l.input_w, 1,
             l.kernel_weights, delta_n, delta_l);
    }
    l.update(l, rate, num, n_delta);
}

void update_connect_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_n = n_delta + offset_o;
        gemm(0, 1, l.output_c, l.output_w,
             l.input_c, l.input_w, 1,
             delta_n, input, l.workspace);
        saxpy_cpu(l.update_kernel_weights, l.workspace, l.output_c * l.input_c, rate, l.update_kernel_weights);
        if (l.bias)
        {
            saxpy_cpu(l.update_bias_weights, delta_n, l.outputs, rate, l.update_bias_weights);
        }
    }
}
