#include "normalization_layer.h"

// Layer *make_normalization_layer()
// {
//     Layer *l = malloc(sizeof(Layer));
//     l->type = NORMALIZE;
//     l->weights = 1;

//     fprintf(stderr, "Normalize       Layer    :    [ksize=%2d]\n");
// }

// void init_normalization_layer(Layer *l, int w, int h, int c)
// {
//     l->input_h = h;
//     l->input_w = w;
//     l->input_c = c;
//     l->inputs = l->input_h * l->input_w * l->input_c;

//     l->output_h = h;
//     l->output_w = w;
//     l->output_c = c;
//     l->outputs = l->output_h * l->output_w * l->output_c;

//     l->workspace_size = 2*l->input_c;

//     l->kernel_weights_size = l->input_c;
//     l->bias_weights_size = l->input_c;
//     l->deltas = l->inputs;

//     if (l->coretype == GPU){
//         l->forward = forward_normalization_layer_gpu;
//         l->backward = backward_normalization_layer_gpu;
//         l->update = update_normalization_layer_gpu;
//     } else {
//         l->forward = forward_normalization_layer;
//         l->backward = backward_normalization_layer;
//         l->update = update_normalization_layer;
//     }
//     fprintf(stderr, "Normalization   Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
//             l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
// }

void forward_normalization_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *input = l.output + offset_o;
        float *output = l.output + offset_o;
        float *norm_x = l.x_norm + offset_o;
        float *mean = l.mean + i*l.output_c;
        float *variance = l.variance + i*l.output_c;
        float *roll_mean = l.rolling_mean + i*l.output_c;
        float *roll_variance = l.rolling_variance + i*l.output_c;
        float *normalize_x = l.normalize_x + i*l.outputs;
        normalize_mean(input, l.output_h, l.output_w, l.output_c, mean);
        normalize_variance(input, l.output_h, l.output_w, l.output_c, mean, variance);
        if (l.train) {
            memcpy(normalize_x, input, l.outputs*sizeof(float));
            normalize_cpu(input, mean, variance, l.output_h, l.output_w, l.output_c, output);
            memcpy(norm_x, output, l.outputs*sizeof(float));
        }
        multy_cpu(mean, l.output_c, .99, 1);
        multy_cpu(variance, l.output_c, .99, 1);
        saxpy_cpu(roll_mean, mean, l.output_c, .01, roll_mean);
        saxpy_cpu(roll_variance, variance, l.output_c, .01, roll_variance);
        if (!l.train) normalize_cpu(input, roll_mean, roll_variance, l.output_h, l.input_w, l.input_c, output);
        scale_bias(output, l.normalize_weights, l.output_c, l.output_h * l.output_w);
        add_bias(output, l.bias_weights, l.output_c, l.output_h * l.output_w);
    }
}

void backward_normalization_layer(Layer l, float rate, int num, float *n_delta)
{
    l.update(l, rate, num, n_delta);
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *input = l.normalize_x + offset_o;
        float *delta_n = n_delta + offset_o;
        float *mean = l.mean + i*l.output_c;
        float *variance = l.variance + i*l.output_c;
        gradient_normalize_mean(l.normalize_weights, variance, l.output_c, l.workspace);
        gradient_normalize_variance(l.normalize_weights, input, n_delta, mean, variance, l.output_h, l.output_w, l.output_c, l.workspace+l.output_c);
        gradient_normalize_cpu(input, mean, l.workspace, l.workspace+l.output_c, l.output_h, l.output_w, l.output_c, delta_n, delta_n, l.workspace+2*l.output_c);
        gradient_normalize_layer(l.output_h, l.output_w, l.output_c, delta_n, l.workspace+2*l.output_c);
    }
}

void update_normalization_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *delta_n = n_delta + offset_o;
        float *norm_x = l.x_norm + offset_o;
        sum_channel_cpu(delta_n, l.output_h, l.output_w, l.output_c, rate, l.workspace);
        add_bias(l.update_bias_weights, l.workspace, l.output_c, l.output_h*l.output_w);
        matrix_multiply_cpu(norm_x, delta_n, l.outputs, l.workspace);
        sum_channel_cpu(l.workspace, l.output_h, l.output_w, l.output_c, rate, l.workspace);
        scale_bias(l.update_normalize_weights, l.workspace, l.output_c, l.output_h*l.output_w);
    }
}
