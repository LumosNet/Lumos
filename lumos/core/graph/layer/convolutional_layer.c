#include "convolutional_layer.h"

Layer *make_convolutional_layer(int filters, int ksize, int stride, int pad, int bias, int normalization, char *active)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = CONVOLUTIONAL;
    l->weights = 1;

    l->filters = filters;
    l->ksize = ksize;
    l->stride = stride;
    l->pad = pad;

    l->bias = bias;

    l->batchnorm = normalization;

    l->active_str = active;
    Activation type = load_activate_type(active);
    l->active = load_activate(type);
    l->gradient = load_gradient(type);

    l->forward = forward_convolutional_layer;
    l->backward = backward_convolutional_layer;

    l->update = update_convolutional_layer;
    l->init_layer_weights = init_convolutional_weights;

    fprintf(stderr, "Convolutional   Layer    :    [filters=%2d, ksize=%2d, stride=%2d, pad=%2d, bias=%d, normalization=%d, active=%s]\n",
            l->filters, l->ksize, l->stride, l->pad, l->bias, l->batchnorm, l->active_str);
    return l;
}

Layer *make_convolutional_layer_by_cfg(CFGParams *p)
{
    int filters = 0;
    int ksize = 0;
    int stride = 0;
    int pad = 0;

    int bias = 0;

    int normalization = 0;

    char *active = NULL;

    CFGParam *param = p->head;
    while (param)
    {
        if (0 == strcmp(param->key, "filters"))
        {
            filters = atoi(param->val);
        }
        else if (0 == strcmp(param->key, "ksize"))
        {
            ksize = atoi(param->val);
        }
        else if (0 == strcmp(param->key, "stride"))
        {
            stride = atoi(param->val);
        }
        else if (0 == strcmp(param->key, "pad"))
        {
            pad = atoi(param->val);
        }
        else if (0 == strcmp(param->key, "bias"))
        {
            bias = atoi(param->val);
        }
        else if (0 == strcmp(param->key, "normalization"))
        {
            normalization = atoi(param->val);
        }
        else if (0 == strcmp(param->key, "active"))
        {
            active = param->val;
        }
        param = param->next;
    }

    Layer *l = make_convolutional_layer(filters, ksize, stride, pad, bias, normalization, active);
    return l;
}

void init_convolutional_layer(Layer *l, int w, int h, int c)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = (l->input_h + 2 * l->pad - l->ksize) / l->stride + 1;
    l->output_w = (l->input_w + 2 * l->pad - l->ksize) / l->stride + 1;
    l->output_c = l->filters;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = l->ksize * l->ksize * l->input_c * l->output_h * l->output_w + l->filters * l->ksize * l->ksize * l->input_c;

    l->kernel_weights_size = l->filters * l->ksize * l->ksize * l->input_c;
    l->bias_weights_size = 0;
    if (l->bias){
        l->bias_weights_size = l->filters;
    }
    l->deltas = l->inputs;

    fprintf(stderr, "Convolutional   Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void init_convolutional_weights(Layer *l)
{
    int offset = 0;
    float *kernel_weights = l->kernel_weights;
    for (int i = 0; i < l->filters; ++i)
    {
        random(1, l->input_h * l->input_w, 0.01, l->ksize * l->ksize, kernel_weights);
        offset += l->ksize * l->ksize;
        for (int j = 0; j < l->input_c - 1; ++j)
        {
            memcpy(kernel_weights + offset, kernel_weights, l->ksize * l->ksize * sizeof(float));
            offset += l->ksize * l->ksize;
        }
        kernel_weights += offset;
        offset = 0;
    }
    for (int i = 0; i < l->bias_weights_size; ++i)
    {
        l->bias_weights[i] = 0.01;
    }
}

void forward_convolutional_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        for (int j = 0; j < l.inputs; ++j){
            printf("%f ", input[j]);
        }
        printf("\n");
        im2col(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
        gemm(0, 0, l.filters, l.ksize * l.ksize * l.input_c, l.ksize * l.ksize * l.input_c, l.output_h * l.output_w, 1,
             l.kernel_weights, l.workspace, output);
        if (l.bias)
        {
            add_bias(output, l.bias_weights, l.filters, l.output_h * l.output_w);
        }
        activate_list(output, l.outputs, l.active);
    }
}

void backward_convolutional_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        gradient_list(output, l.outputs, l.gradient);
        multiply(delta_n, output, l.outputs, delta_n);
        gemm(1, 0, l.filters, l.ksize * l.ksize * l.input_c,
             l.filters, l.output_h * l.output_w, 1,
             l.kernel_weights, delta_n, l.workspace);
        col2im(l.workspace, l.ksize, l.stride, l.pad, l.input_h, l.input_w, l.input_c, delta_l);
    }
    l.update(l, rate, num, n_delta);
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
        saxpy(l.update_kernel_weights, l.workspace + l.ksize * l.ksize * l.input_c * l.output_h * l.output_w, l.filters * l.ksize * l.ksize * l.input_c, rate, l.update_kernel_weights);
        if (l.bias)
        {
            for (int j = 0; j < l.filters; ++j)
            {
                float bias = sum_cpu(delta_n, l.output_h * l.output_w);
                l.update_bias_weights[j] += bias * rate;
            }
        }
    }
}
