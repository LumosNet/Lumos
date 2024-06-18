#include "normalization_layer.h"

void init_normalization_layer(Layer *l, int subdivision)
{
    l->mean = calloc(l->output_c, sizeof(float));
    l->variance = calloc(l->output_c, sizeof(float));
    l->rolling_mean = calloc(l->output_c, sizeof(float));
    l->rolling_variance = calloc(l->output_c, sizeof(float));
    l->normalize_x = calloc(subdivision*l->outputs, sizeof(float));
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
    normalize_mean(l.output, l.output_h, l.output_w, l.output_c, num, l.mean);
    normalize_variance(l.output, l.output_h, l.output_w, l.output_c, num, l.mean, l.variance);
    multy_cpu(l.rolling_mean, l.output_c, .99, 1);
    multy_cpu(l.rolling_variance, l.output_c, .99, 1);
    saxpy_cpu(l.rolling_mean, l.mean, l.output_c, .01, l.rolling_mean);
    saxpy_cpu(l.rolling_variance, l.variance, l.output_c, .01, l.rolling_variance);
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *input = l.output + offset_o;
        float *output = l.output + offset_o;
        float *norm_x = l.x_norm + offset_o;
        float *normalize_x = l.normalize_x + offset_o;
        if (l.status) {
            memcpy(normalize_x, input, l.outputs*sizeof(float));
            normalize_cpu(input, l.mean, l.variance, l.output_h, l.output_w, l.output_c, output);
            memcpy(norm_x, output, l.outputs*sizeof(float));
        }
        if (!l.status) normalize_cpu(input, l.rolling_mean, l.rolling_variance, l.output_h, l.output_w, l.output_c, output);
        scale_bias(output, l.bn_scale, l.output_c, l.output_h * l.output_w);
        add_bias(output, l.bn_bias, l.output_c, l.output_h * l.output_w);
    }
}

void backward_normalization_layer(Layer l, float rate, int num, float *n_delta)
{
    update_normalization_layer(l, rate, num, n_delta);
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *input = l.normalize_x + offset_o;
        float *delta_n = n_delta + offset_o;
        scale_bias(delta_n, l.bn_scale, l.output_c, l.output_h*l.output_w);
        gradient_normalize_mean(delta_n, l.variance, l.output_h, l.output_w, l.output_c, l.mean_delta);
        gradient_normalize_variance(delta_n, input, l.mean, l.variance, l.output_h, l.output_w, l.output_c, l.variance_delta);
        gradient_normalize_cpu(input, l.mean, l.variance, l.mean_delta, l.variance_delta, l.output_h, l.output_w, l.output_c, delta_n, delta_n);
    }
}

void update_normalization_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *delta_n = n_delta + offset_o;
        float *norm_x = l.x_norm + offset_o;
        update_scale(norm_x, delta_n, l.output_h, l.output_w, l.output_c, rate, l.update_bn_scale);
        update_bias(delta_n, l.output_h, l.output_w, l.output_c, rate, l.update_bn_bias);
    }
}

void update_normalization_layer_weights(Layer l)
{
    memcpy(l.bn_scale, l.update_bn_scale, l.output_c*sizeof(float));
    memcpy(l.bn_bias, l.update_bn_bias, l.output_c*sizeof(float));
}

void save_normalization_layer_weights(Layer l, FILE *fp)
{
    fwrite(l.bn_scale, sizeof(float), l.output_c, fp);
    fwrite(l.bn_bias, sizeof(float), l.output_c, fp);
    fwrite(l.rolling_mean, sizeof(float), l.output_c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.output_c, fp);
}