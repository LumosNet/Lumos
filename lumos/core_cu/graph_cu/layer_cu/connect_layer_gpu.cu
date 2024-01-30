#include "connect_layer_gpu.h"

void init_connect_layer_gpu(Layer *l, int w, int h, int c)
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

    cudaMalloc((void**)&l->output, l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, l->inputs*sizeof(float));
    cudaMalloc((void**)&l->kernel_weights, l->inputs*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->update_kernel_weights, l->inputs*l->outputs*sizeof(float));
    if (l->bias){
        cudaMalloc((void**)&l->bias_weights, l->outputs*sizeof(float));
        cudaMalloc((void**)&l->update_bias_weights, l->outputs*sizeof(float));
    }

    fprintf(stderr, "Connect         Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void weightinit_connect_layer_gpu(Layer l)
{
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
    float *bias_weights = NULL;
    float scale = sqrt((float)2 / l.inputs);
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        kernel_weights[i] = scale*rand_uniform(-1, 1);
    }
    if (l.bias){
        bias_weights = (float*)calloc(l.outputs, sizeof(float));
        fill_cpu(bias_weights, l.outputs, 0.001, 1);
        cudaMemcpy(l.bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.update_bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
        free(bias_weights);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void forward_connect_layer_gpu(Layer l, int num)
{
    // float *input_t = (float*)calloc(l.inputs*num, sizeof(float));
    // cudaMemcpy(input_t, l.input, l.inputs*num*sizeof(float), cudaMemcpyDeviceToHost);
    // printf("input: -------\n");
    // for (int i = 0; i < l.inputs*num; ++i){
    //     printf("%f ", input_t[i]);
    // }
    // printf("\n");
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
        activate_list_gpu(output, l.outputs, l.activegpu);
    }
    // float *output_t = (float*)calloc(l.outputs*num, sizeof(float));
    // cudaMemcpy(output_t, l.output, l.outputs*num*sizeof(float), cudaMemcpyDeviceToHost);
    // printf("output: -------\n");
    // for (int i = 0; i < l.outputs*num; ++i){
    //     printf("%f ", output_t[i]);
    // }
    // printf("\n");
}

void backward_connect_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        gradient_list_gpu(output, l.outputs, l.gradientgpu);
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
