#include "normalization_layer_gpu.h"

void init_normalization_layer_gpu(Layer *l, int subdivision)
{
    cudaMalloc((void**)&l->mean, l->output_c*sizeof(float));
    cudaMalloc((void**)&l->variance, l->output_c*sizeof(float));
    cudaMalloc((void**)&l->rolling_mean, l->output_c*sizeof(float));
    cudaMalloc((void**)&l->rolling_variance, l->output_c*sizeof(float));
    cudaMalloc((void**)&l->normalize_x, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->x_norm, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->mean_delta, l->output_c*sizeof(float));
    cudaMalloc((void**)&l->variance_delta, l->output_c*sizeof(float));
    cudaMalloc((void**)&l->bn_scale, l->output_c*sizeof(float));
    cudaMalloc((void**)&l->bn_bias, l->output_c*sizeof(float));
    cudaMalloc((void**)&l->update_bn_scale, l->output_c*sizeof(float));
    cudaMalloc((void**)&l->update_bn_bias, l->output_c*sizeof(float));
}

void weightinit_normalization_layer_gpu(Layer l, FILE *fp)
{
    if (fp){
        float *bn_scale = (float*)calloc(l.output_c, sizeof(float));
        float *bn_bias = (float*)calloc(l.output_c, sizeof(float));
        float *rolling_mean = (float*)calloc(l.output_c, sizeof(float));
        float *rolling_variance = (float*)calloc(l.output_c, sizeof(float));
        fread(bn_scale, sizeof(float), l.output_c, fp);
        fread(bn_bias, sizeof(float), l.output_c, fp);
        fread(rolling_mean, sizeof(float), l.output_c, fp);
        fread(rolling_variance, sizeof(float), l.output_c, fp);
        cudaMemcpy(l.bn_scale, bn_scale, l.output_c*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.bn_bias, bn_bias, l.output_c*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.rolling_mean, rolling_mean, l.output_c*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.rolling_variance, rolling_variance, l.output_c*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.update_bn_scale, bn_scale, l.output_c*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.update_bn_bias, bn_bias, l.output_c*sizeof(float), cudaMemcpyHostToDevice);
        free(bn_scale);
        free(bn_bias);
        free(rolling_mean);
        free(rolling_variance);
        return;
    }
    fill_gpu(l.bn_scale, l.output_c, 1, 1);
    fill_gpu(l.update_bn_scale, l.output_c, 1, 1);
    fill_gpu(l.bn_bias, l.output_c, 0, 1);
    fill_gpu(l.update_bn_bias, l.output_c, 0, 1);
}

void forward_normalization_layer_gpu(Layer l, int num)
{
    normalize_mean_gpu(l.output, l.output_h, l.output_w, l.output_c, num, l.mean);
    normalize_variance_gpu(l.output, l.output_h, l.output_w, l.output_c, num, l.mean, l.variance);
    multy_gpu(l.rolling_mean, l.output_c, .99, 1);
    multy_gpu(l.rolling_variance, l.output_c, .99, 1);
    saxpy_gpu(l.rolling_mean, l.mean, l.output_c, .01, l.rolling_mean);
    saxpy_gpu(l.rolling_variance, l.variance, l.output_c, .01, l.rolling_variance);
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *input = l.output + offset_o;
        float *output = l.output + offset_o;
        float *norm_x = l.x_norm + offset_o;
        float *normalize_x = l.normalize_x + offset_o;
        if (l.status) {
            cudaMemcpy(normalize_x, input, l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
            normalize_gpu(input, l.mean, l.variance, l.output_h, l.output_w, l.output_c, output);
            cudaMemcpy(norm_x, output, l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
        }
        if (!l.status) normalize_gpu(input, l.rolling_mean, l.rolling_variance, l.output_h, l.output_w, l.output_c, output);
        scale_bias_gpu(output, l.bn_scale, l.output_c, l.output_h * l.output_w);
        add_bias_gpu(output, l.bn_bias, l.output_c, l.output_h * l.output_w);
    }
}

void backward_normalization_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    update_normalization_layer_gpu(l, rate, num, n_delta);
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *input = l.normalize_x + offset_o;
        float *delta_n = n_delta + offset_o;
        scale_bias_gpu(delta_n, l.bn_scale, l.output_c, l.output_h*l.output_w);
        gradient_normalize_mean_gpu(delta_n, l.variance, l.output_h, l.output_w, l.output_c, l.mean_delta);
        gradient_normalize_variance_gpu(delta_n, input, l.mean, l.variance, l.output_h, l.output_w, l.output_c, l.variance_delta);
        gradient_normalize_gpu(input, l.mean, l.variance, l.mean_delta, l.variance_delta, l.output_h, l.output_w, l.output_c, delta_n, delta_n);
    }
}

void update_normalization_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *delta_n = n_delta + offset_o;
        float *norm_x = l.x_norm + offset_o;
        update_scale_gpu(norm_x, delta_n, l.output_h, l.output_w, l.output_c, rate, l.update_bn_scale);
        update_bias_gpu(delta_n, l.output_h, l.output_w, l.output_c, rate, l.update_bn_bias);
    }
}

void update_normalization_layer_weights_gpu(Layer l)
{
    cudaMemcpy(l.bn_scale, l.update_bn_scale, l.output_c*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(l.bn_bias, l.update_bn_bias, l.output_c*sizeof(float), cudaMemcpyDeviceToDevice);
}

void save_normalization_layer_weights_gpu(Layer l, FILE *fp)
{
    float *bn_scale = (float*)calloc(l.output_c, sizeof(float));
    float *bn_bias = (float*)calloc(l.output_c, sizeof(float));
    float *rolling_mean = (float*)calloc(l.output_c, sizeof(float));
    float *rolling_variance = (float*)calloc(l.output_c, sizeof(float));
    cudaMemcpy(bn_scale, l.bn_scale, l.output_c*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bn_bias, l.bn_bias, l.output_c*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rolling_mean, l.rolling_mean, l.output_c*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rolling_variance, l.rolling_variance, l.output_c*sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(bn_scale, sizeof(float), l.output_c, fp);
    fwrite(bn_bias, sizeof(float), l.output_c, fp);
    fwrite(rolling_mean, sizeof(float), l.output_c, fp);
    fwrite(rolling_variance, sizeof(float), l.output_c, fp);
    free(bn_scale);
    free(bn_bias);
    free(rolling_mean);
    free(rolling_variance);
}