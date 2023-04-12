#include "im2col_layer_gpu.h"

void forward_im2col_layer_gpu(Layer l, int num)
{
    cudaMemcpy(l.output, l.input, num*l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
    float output_cpu[l.outputs];
    cudaMemcpy(output_cpu, l.output, l.outputs*sizeof(float), cudaMemcpyDeviceToHost);
}

void backward_im2col_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    cudaMemcpy(l.delta, n_delta, num*l.inputs*sizeof(float), cudaMemcpyDeviceToDevice);
}
