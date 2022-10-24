#include "avgpool_layer_gpu.h"

__global__ void avgpool_kernel()
{

}

void forward_avgpool_layer_gpu(Layer l, int num)
{
    int size = l.input_h*l.input_w;
    dim3 dimGrid(, l.outputs, );
    dim3 dimBlock(BLOCK, 1, 1);
    fill_gpu(l.output, l.outputs * num, 0, 1);

}
