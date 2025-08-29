#include "connect_layer_gpu.h"

void init_connect_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
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

    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));
    cudaMalloc((void**)&l->kernel_weights, l->inputs*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->update_kernel_weights, l->inputs*l->outputs*sizeof(float));
    if (l->bias){
        cudaMalloc((void**)&l->bias_weights, l->outputs*sizeof(float));
        cudaMalloc((void**)&l->update_bias_weights, l->outputs*sizeof(float));
    }

    fprintf(stderr, "Connect         Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void weightinit_connect_layer_gpu(Layer l, FILE *fp)
{
    if (fp){
        float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
        fread(kernel_weights, sizeof(float), l.outputs*l.inputs, fp);
        cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
        free(kernel_weights);
        if (l.bias){
            float *bias_weights = (float*)calloc(l.outputs, sizeof(float));
            fread(bias_weights, sizeof(float), l.outputs, fp);
            cudaMemcpy(l.bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(l.update_bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
            free(bias_weights);
        }
        return;
    }
    InitCpt initcpt = *l.initcpt;
    if (initcpt.initype == CONSTANT_I) connect_constant_init_gpu(l, initcpt.x);
    else if (initcpt.initype == NORMAL_I) connect_normal_init_gpu(l, initcpt.mean, initcpt.std);
    else if (initcpt.initype == UNIFORM_I) connect_uniform_init_gpu(l, initcpt.min, initcpt.max);
    else if (initcpt.initype == KAIMING_NORMAL_I) connect_kaiming_normal_init_gpu(l, initcpt.a, initcpt.mode, initcpt.nonlinearity);
    else if (initcpt.initype == KAIMING_UNIFORM_I) connect_kaiming_uniform_init_gpu(l, initcpt.a, initcpt.mode, initcpt.nonlinearity);
    else connect_constant_init_gpu(l, 0);
    if (l.bias){
        float *bias_weights = (float*)calloc(l.outputs, sizeof(float));
        fill_cpu(bias_weights, l.outputs, 0.001, 1);
        cudaMemcpy(l.bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.update_bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
        free(bias_weights);
    }
}

void forward_connect_layer_gpu(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
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
        gradient_list_gpu(output, l.outputs, l.active);
        matrix_multiply_gpu(delta_n, output, l.outputs, delta_n);
        gemm_gpu(1, 0, l.output_c, l.input_c, l.output_c, l.input_w, 1,
             l.kernel_weights, delta_n, delta_l);
    }
    update_connect_layer_gpu(l, rate, num, n_delta);
}

void update_connect_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_n = n_delta + offset_o;
        gemm_gpu(0, 1, l.output_c, l.output_w,
             l.input_c, l.input_w, 1,
             delta_n, input, l.workspace);
        saxpy_gpu(l.update_kernel_weights, l.workspace, l.output_c * l.input_c, rate, l.update_kernel_weights);
        if (l.bias)
        {
            saxpy_gpu(l.update_bias_weights, delta_n, l.outputs, rate, l.update_bias_weights);
        }
    }
}

void update_connect_layer_weights_gpu(Layer l)
{
    cudaMemcpy(l.kernel_weights, l.update_kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
    if (l.bias){
        cudaMemcpy(l.bias_weights, l.update_bias_weights, l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

void save_connect_layer_weights_gpu(Layer l, FILE *fp)
{
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
    cudaMemcpy(kernel_weights, l.kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(kernel_weights, sizeof(float), l.inputs*l.outputs, fp);
    free(kernel_weights);
    if (l.bias){
        float *bias_weights = (float*)calloc(l.outputs, sizeof(float));
        cudaMemcpy(bias_weights, l.bias_weights, l.outputs*sizeof(float), cudaMemcpyDeviceToHost);
        fwrite(bias_weights, sizeof(float), l.outputs, fp);
        free(bias_weights);
    }
}

void free_connect_layer_gpu(Layer l)
{
    cudaFree(l.output);
    cudaFree(l.delta);
    cudaFree(l.kernel_weights);
    cudaFree(l.update_kernel_weights);
    if (l.bias){
        cudaFree(l.bias_weights);
        cudaFree(l.update_bias_weights);
    }
}

void connect_constant_init_gpu(Layer l, float x)
{
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        kernel_weights[i] = x;
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void connect_normal_init_gpu(Layer l, float mean, float std)
{
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        kernel_weights[i] = generate_normal(mean, std);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void connect_uniform_init_gpu(Layer l, float min, float max)
{
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        kernel_weights[i] = rand_uniform(min, max);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void connect_kaiming_normal_init_gpu(Layer l, float a, char *mode, char *nonlinearity)
{
    if (0 == strcmp(nonlinearity, "relu")) a = 0;
    else if (0 == strcmp(nonlinearity, "leaky relu")) a = 0.1;
    else a = 0;
    int num = 0;
    if (0 == strcmp(mode, "fan_in")) num = l.inputs;
    else if (0 == strcmp(mode, "fan_out")) num = l.outputs;
    else num = l.inputs;
    float scale = sqrt((float)2/(1+a*a)*num);
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        kernel_weights[i] = scale*generate_normal(0, 1);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void connect_kaiming_uniform_init_gpu(Layer l, float a, char *mode, char *nonlinearity)
{
    if (0 == strcmp(nonlinearity, "relu")) a = 0;
    else if (0 == strcmp(nonlinearity, "leaky relu")) a = 0.1;
    else a = 0;
    int num = 0;
    if (0 == strcmp(mode, "fan_in")) num = l.inputs;
    else if (0 == strcmp(mode, "fan_out")) num = l.outputs;
    else num = l.inputs;
    float scale = sqrt((float)2/(1+a*a)*num);
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        kernel_weights[i] = scale*rand_uniform(-1, 1);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}
