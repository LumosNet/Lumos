#include "normalization_layer.h"

Layer *make_normalization_layer()
{
    Layer *l = malloc(sizeof(Layer));
    l->type = NORMALIZE;
    l->weights = 0;

    l->update = NULL;

    fprintf(stderr, "Normalize       Layer    :    [ksize=%2d]\n");
}

void init_normalization_layer(Layer *l, int w, int h, int c)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = h;
    l->output_w = w;
    l->output_c = c;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = 0;
    l->deltas = l->inputs;

    if (l->coretype == GPU){
        l->forward = forward_normalization_layer_gpu;
        l->backward = backward_normalization_layer_gpu;
    } else {
        l->forward = forward_normalization_layer;
        l->backward = backward_normalization_layer;
    }
}

void forward_normalization_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        float *mean = l.mean + i*l.input_c;
        float *variance = l.variance + i*l.input_c;
        float *roll_mean = l.rolling_mean + i*l.input_c;
        float *roll_variance = l.rooling_variance + i*l.input_c;
        normalize_mean(input, l.input_h, l.input_w, l.input_c, mean);
        normalize_variance(input, l.input_h, l.input_w, l.input_c, mean, variance);
        if (l.train) normalize_cpu(input, mean, variance, l.input_h, l.input_w, l.input_c, output);
        multy_cpu(mean, l.input_c, .99, 1);
        multy_cpu(variance, l.input_c, .99, 1);
        saxpy_cpu(roll_mean, mean, l.input_c, .01, roll_mean);
        saxpy_cpu(roll_variance, variance, l.input_c, .01, roll_variance);
        if (!l.train) normalize_cpu(input, roll_mean, roll_variance, l.input_h, l.input_w, l.input_c, output);
        scale_bias(output, l.normalize_scale, l.input_c, l.output_h * l.output_w);
        add_bias(output, l.normalize_bias, l.input_c, l.output_h * l.output_w);
    }
}

void backward_normalization_layer(Layer l, float rate, int num, float *n_delta)
{

}
