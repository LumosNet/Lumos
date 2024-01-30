#include "im2col_layer_gpu.h"

void init_im2col_layer_gpu(Layer *l, int w, int h, int c)
{
    l->input_h = h,
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = 1;
    l->output_w = 1;
    l->output_c = l->inputs;
    l->outputs = l->inputs;

    l->workspace_size = 0;
    cudaMalloc((void**)&l->output, l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, l->inputs*sizeof(float));

    fprintf(stderr, "Im2col          Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_im2col_layer_gpu(Layer l, int num)
{
    // float *input_t = (float*)calloc(l.inputs*num, sizeof(float));
    // cudaMemcpy(input_t, l.input, l.inputs*num*sizeof(float), cudaMemcpyDeviceToHost);
    // printf("input: -------\n");
    // for (int i = 0; i < l.inputs*num; ++i){
    //     printf("%f ", input_t[i]);
    // }
    // printf("\n");
    cudaMemcpy(l.output, l.input, num*l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
    // float *output_t = (float*)calloc(l.outputs*num, sizeof(float));
    // cudaMemcpy(output_t, l.output, l.outputs*num*sizeof(float), cudaMemcpyDeviceToHost);
    // printf("output: -------\n");
    // for (int i = 0; i < l.outputs*num; ++i){
    //     printf("%f ", output_t[i]);
    // }
    // printf("\n");
}

void backward_im2col_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    cudaMemcpy(l.delta, n_delta, num*l.inputs*sizeof(float), cudaMemcpyDeviceToDevice);
}
