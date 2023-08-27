#include "normalization_layer_gpu.h"

void forward_normalization_layer_gpu(Layer l, int num)
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
        normalize_mean_gpu(input, l.output_h, l.output_w, l.output_c, mean);
        normalize_variance_gpu(input, l.output_h, l.output_w, l.output_c, mean, variance);
        if (l.train) {
            cudaMemcpy(normalize_x, input, l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
            normalize_gpu(input, mean, variance, l.output_h, l.output_w, l.output_c, output);
            cudaMemcpy(norm_x, output, l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
        }
        multy_gpu(mean, l.output_c, .99, 1);
        multy_gpu(variance, l.output_c, .99, 1);
        saxpy_gpu(roll_mean, mean, l.output_c, .01, roll_mean);
        saxpy_gpu(roll_variance, variance, l.output_c, .01, roll_variance);
        if (!l.train) normalize_gpu(input, roll_mean, roll_variance, l.output_h, l.input_w, l.input_c, output);
        scale_bias_gpu(output, l.kernel_weights, l.output_c, l.output_h * l.output_w);
        add_bias_gpu(output, l.bias_weights, l.output_c, l.output_h * l.output_w);
    }
}

void backward_normalization_layer_gpu(Layer l, float rate, int num)
{
    update_normalization_layer_gpu(l, rate, num);
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *input = l.normalize_x + offset_o;
        float *delta_n = l.n_delta + offset_o;
        float *mean = l.mean + i*l.output_c;
        float *variance = l.variance + i*l.output_c;
        gradient_normalize_mean_gpu(l.normalize_weights, variance, l.output_c, l.workspace);
        gradient_normalize_variance_gpu(l.normalize_weights, input, l.n_delta, mean, variance, l.output_h, l.output_w, l.output_c, l.workspace+l.output_c);
        gradient_normalize_gpu(input, mean, l.workspace, l.workspace+l.output_c, l.output_h, l.output_w, l.output_c, delta_n, delta_n, l.workspace+2*l.output_c);
        gradient_normalize_layer_gpu(l.output_h, l.output_w, l.output_c, delta_n, l.workspace+2*l.output_c);
    }
}

void update_normalization_layer_gpu(Layer l, float rate, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *delta_n = l.n_delta + offset_o;
        float *norm_x = l.x_norm + offset_o;
        sum_channel_gpu(delta_n, l.output_h, l.output_w, l.output_c, rate, l.workspace);
        add_bias_gpu(l.update_bias_weights, l.workspace, l.output_c, l.output_h*l.output_w);
        matrix_multiply_gpu(norm_x, delta_n, l.outputs, l.workspace);
        sum_channel_gpu(l.workspace, l.output_h, l.output_w, l.output_c, rate, l.workspace);
        scale_bias_gpu(l.update_normalize_weights, l.workspace, l.output_c, l.output_h*l.output_w);
    }
}
