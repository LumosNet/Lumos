#include "normalization_layer.h"

void init_normalization_layer(Layer *l, int w, int h, int c, int subdivision)
{
    l->mean = calloc(l->output_c, sizeof(float));
    l->variance = calloc(l->output_c, sizeof(float));
    l->rolling_mean = calloc(l->output_c, sizeof(float));
    l->rolling_variance = calloc(l->output_c, sizeof(float));
    l->normalize_x = calloc(subdivision*l->inputs, sizeof(float));
    l->x_norm = calloc(subdivision*l->outputs, sizeof(float));
    l->mean_delta = calloc(l->output_c, sizeof(float));
    l->variance_delta = calloc(l->output_c, sizeof(float));
    l->bn_scale = calloc(l->output_c, sizeof(float));
    l->bn_bias = calloc(l->output_c, sizeof(float));
    l->update_bn_scale = calloc(l->output_c, sizeof(float));
    l->update_bn_bias = calloc(l->output_c, sizeof(float));
}

void weightinit_normalization_layer(Layer l, FILE *fp)
{
    if (fp){
        fread(l.bn_scale, sizeof(float), l.output_c, fp);
        fread(l.bn_bias, sizeof(float), l.output_c, fp);
        fread(l.rolling_mean, sizeof(float), l.output_c, fp);
        fread(l.rolling_variance, sizeof(float), l.output_c, fp);
        memcpy(l.update_bn_scale, l.bn_scale, l.output_c*sizeof(float));
        memcpy(l.update_bn_bias, l.bn_bias, l.output_c*sizeof(float));
        return;
    }
    fill_cpu(l.bn_scale, l.output_c, 1, 1);
    fill_cpu(l.update_bn_scale, l.output_c, 1, 1);
}

void forward_normalization_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *input = l.output + offset_o;
        float *output = l.output + offset_o;
        float *norm_x = l.x_norm + offset_o;
        float *normalize_x = l.normalize_x + offset_o;
        float *mean = l.mean + i*l.output_c;
        float *variance = l.variance + i*l.output_c;
        float *roll_mean = l.rolling_mean + i*l.output_c;
        float *roll_variance = l.rolling_variance + i*l.output_c;
        normalize_mean(input, l.output_h, l.output_w, l.output_c, mean);
        normalize_variance(input, l.output_h, l.output_w, l.output_c, mean, variance);
        if (l.status) {
            memcpy(normalize_x, input, l.outputs*sizeof(float));
            normalize_cpu(input, mean, variance, l.output_h, l.output_w, l.output_c, output);
            memcpy(norm_x, output, l.outputs*sizeof(float));
        }
        multy_cpu(roll_mean, l.output_c, .99, 1);
        multy_cpu(roll_variance, l.output_c, .99, 1);
        saxpy_cpu(roll_mean, mean, l.output_c, .01, roll_mean);
        saxpy_cpu(roll_variance, variance, l.output_c, .01, roll_variance);
        if (!l.status) normalize_cpu(input, roll_mean, roll_variance, l.output_h, l.input_w, l.input_c, output);
        scale_bias(output, l.bn_scale, l.output_c, l.output_h * l.output_w);
        add_bias(output, l.bn_bias, l.output_c, l.output_h * l.output_w);
    }
}

void backward_normalization_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *input = l.normalize_x + offset_o;
        float *delta_n = n_delta + offset_o;
        float *mean = l.mean + i*l.output_c;
        float *variance = l.variance + i*l.output_c;
        gradient_normalize_mean(l.bn_scale, variance, l.output_c, l.workspace);
        gradient_normalize_variance(l.bn_scale, input, n_delta, mean, variance, l.output_h, l.output_w, l.output_c, l.workspace+l.output_c);
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

void save_normalization_layer_weights(Layer l, FILE *fp)
{
    fwrite(l.bn_scale, sizeof(float), l.output_c, fp);
    fwrite(l.bn_bias, sizeof(float), l.output_c, fp);
    fwrite(l.rolling_mean, sizeof(float), l.output_c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.output_c, fp);
}