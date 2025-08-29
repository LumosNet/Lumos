#include "convolutional_layer_gpu.h"

void init_convolutional_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
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

    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));
    cudaMalloc((void**)&l->kernel_weights, l->filters*l->ksize*l->ksize*l->input_c*sizeof(float));
    cudaMalloc((void**)&l->update_kernel_weights, l->filters*l->ksize*l->ksize*l->input_c*sizeof(float));
    if (l->bias){
        cudaMalloc((void**)&l->bias_weights, l->filters*sizeof(float));
        cudaMalloc((void**)&l->update_bias_weights, l->filters*sizeof(float));
    }
    if (l->normalize) init_normalization_layer_gpu(l, subdivision);

    fprintf(stderr, "Convolutional   Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void weightinit_convolutional_layer_gpu(Layer l, FILE *fp)
{
    if (fp){
        float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c, sizeof(float));
        fread(kernel_weights, sizeof(float), l.ksize*l.ksize*l.filters*l.input_c, fp);
        cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
        free(kernel_weights);
        if (l.bias){
            float *bias_weights = (float*)calloc(l.filters, sizeof(float));
            fread(bias_weights, sizeof(float), l.filters, fp);
            cudaMemcpy(l.bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(l.update_bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
            free(bias_weights);
        }
        return;
    }
    InitCpt initcpt = *l.initcpt;
    if (initcpt.initype == CONSTANT_I) convolutional_constant_init_gpu(l, initcpt.x);
    else if (initcpt.initype == NORMAL_I) convolutional_normal_init_gpu(l, initcpt.mean, initcpt.std);
    else if (initcpt.initype == KAIMING_NORMAL_I) convolutional_kaiming_normal_init_gpu(l, initcpt.a, initcpt.mode, initcpt.nonlinearity);
    else convolutional_constant_init_gpu(l, 0);
    if (l.bias){
        float *bias_weights = (float*)calloc(l.filters, sizeof(float));
        fill_cpu(bias_weights, l.filters, 0.001, 1);
        cudaMemcpy(l.bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.update_bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
        free(bias_weights);
    }
}

void forward_convolutional_layer_gpu(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        im2col_gpu(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
        gemm_gpu(0, 0, l.filters, l.ksize * l.ksize * l.input_c, l.ksize * l.ksize * l.input_c, l.output_h * l.output_w, 1,
             l.kernel_weights, l.workspace, output);
        if (l.bias){
            add_bias_gpu(output, l.bias_weights, l.filters, l.output_h * l.output_w);
        }
        if (l.normalize){
            forward_normalization_layer_gpu(l, num);
        }
        activate_list_gpu(output, l.outputs, l.active);
    }
}

void backward_convolutional_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    if (l.normalize){
        backward_normalization_layer_gpu(l, rate, num, n_delta);
    }
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        gradient_list_gpu(output, l.outputs, l.active);
        matrix_multiply_gpu(delta_n, output, l.outputs, delta_n);
        gemm_gpu(1, 0, l.filters, l.ksize * l.ksize * l.input_c,
             l.filters, l.output_h * l.output_w, 1,
             l.kernel_weights, delta_n, l.workspace);
        col2im_gpu(l.workspace, l.ksize, l.stride, l.pad, l.input_h, l.input_w, l.input_c, delta_l);
    }
    update_convolutional_layer_gpu(l, rate, num, n_delta);
}

void update_convolutional_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_n = n_delta + offset_o;
        im2col_gpu(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
        gemm_gpu(0, 1, l.filters, l.output_h * l.output_w,
             l.ksize * l.ksize * l.input_c, l.output_h * l.output_w, 1,
             delta_n, l.workspace, l.workspace + l.ksize * l.ksize * l.input_c * l.output_h * l.output_w);
        saxpy_gpu(l.update_kernel_weights, l.workspace + l.ksize * l.ksize * l.input_c * l.output_h * l.output_w, l.filters * l.ksize * l.ksize * l.input_c, rate, l.update_kernel_weights);
        if (l.bias){
            sum_channel_gpu(delta_n, l.output_h, l.output_w, l.output_c, rate, l.workspace);
            add_bias_gpu(l.update_bias_weights, l.workspace, l.output_c, 1);
        }
    }
}

void update_convolutional_layer_weights_gpu(Layer l)
{
    cudaMemcpy(l.kernel_weights, l.update_kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyDeviceToDevice);
    if (l.bias){
        cudaMemcpy(l.bias_weights, l.update_bias_weights, l.filters*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    if (l.normalize) update_normalization_layer_weights_gpu(l);
}

void save_convolutional_layer_weights_gpu(Layer l, FILE *fp)
{
    float *kernel_weights = (float*)calloc(l.ksize*l.ksize*l.filters*l.input_c, sizeof(float));
    cudaMemcpy(kernel_weights, l.kernel_weights, l.ksize*l.ksize*l.filters*l.input_c*sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(kernel_weights, sizeof(float), l.ksize*l.ksize*l.filters*l.input_c, fp);
    free(kernel_weights);
    if (l.bias){
        float *bias_weights = (float*)calloc(l.filters, sizeof(float));
        cudaMemcpy(bias_weights, l.bias_weights, l.filters*sizeof(float), cudaMemcpyDeviceToHost);
        fwrite(bias_weights, sizeof(float), l.filters, fp);
        free(bias_weights);
    }
    if (l.normalize){
        save_normalization_layer_weights_gpu(l, fp);
    }
}

void free_convolutional_layer_gpu(Layer l)
{
    cudaFree(l.output);
    cudaFree(l.delta);
    cudaFree(l.kernel_weights);
    cudaFree(l.update_kernel_weights);
    if (l.bias){
        cudaFree(l.bias_weights);
        cudaFree(l.update_bias_weights);
    }
    if (l.normalize){
        free_normalization_layer_gpu(l);
    }
}

void convolutional_constant_init_gpu(Layer l, float x)
{
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c, sizeof(float));
    for (int i = 0; i < l.filters; ++i){
        float *weight = kernel_weights + i*l.input_c*l.ksize*l.ksize;
        for (int j = 0; j < l.ksize*l.ksize; ++j){
            weight[j] = x;
        }
        for (int j = 0; j < l.input_c-1; ++j){
            float *weight_c = weight + (j+1)*l.ksize*l.ksize;
            memcpy(weight_c, weight, l.ksize*l.ksize*sizeof(float));
        }
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void convolutional_normal_init_gpu(Layer l, float mean, float std)
{
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c, sizeof(float));
    for (int i = 0; i < l.filters; ++i){
        float *weight = kernel_weights + i*l.input_c*l.ksize*l.ksize;
        for (int j = 0; j < l.ksize*l.ksize; ++j){
            weight[j] = generate_normal(mean, std);
        }
        for (int j = 0; j < l.input_c-1; ++j){
            float *weight_c = weight + (j+1)*l.ksize*l.ksize;
            memcpy(weight_c, weight, l.ksize*l.ksize*sizeof(float));
        }
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void convolutional_kaiming_normal_init_gpu(Layer l, float a, char *mode, char *nonlinearity)
{
    if (0 == strcmp(nonlinearity, "relu")) a = 0;
    else if (0 == strcmp(nonlinearity, "leaky relu")) a = 0.1;
    else a = 0;
    int num = 0;
    if (0 == strcmp(mode, "fan_in")) num = l.ksize*l.ksize*l.input_c;
    else if (0 == strcmp(mode, "fan_out")) num = l.ksize*l.ksize*l.output_c;
    else num = l.ksize*l.ksize*l.input_c;
    float scale = sqrt((float)2/(1+a*a)*num);
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c, sizeof(float));
    for (int i = 0; i < l.filters; ++i){
        float *weight = kernel_weights + i*l.input_c*l.ksize*l.ksize;
        for (int j = 0; j < l.ksize*l.ksize; ++j){
            weight[j] = scale*rand_normal();
        }
        for (int j = 0; j < l.input_c-1; ++j){
            float *weight_c = weight + (j+1)*l.ksize*l.ksize;
            memcpy(weight_c, weight, l.ksize*l.ksize*sizeof(float));
        }
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}
