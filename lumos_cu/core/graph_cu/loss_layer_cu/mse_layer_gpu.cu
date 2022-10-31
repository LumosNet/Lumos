#include "mse_layer_gpu.h"

void forward_mse_layer_gpu(Layer l, int num)
{
    float loss = 0;
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        int offset_t = i*l.group;
        float *input = l.input+offset_i;
        float *output = l.output+offset_o;
        float *truth = l.truth+offset_t;
        matrix_subtract_cpu(truth, input, l.inputs, l.workspace);
        gemm(1, 0, l.input_h, l.input_w, l.input_h, l.input_w, 1, \
            l.workspace, l.workspace, output);
        multy_cpu(output, l.outputs, 1/(float)l.group, 1);
        loss += output[0];
    }
    l.loss[0] = loss / num;
}

void backward_mse_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_t = i*l.group;
        float *input = l.input+offset_i;
        float *delta_l = l.delta+offset_i;
        float *truth = l.truth+offset_t;
        matrix_subtract_cpu(input, truth, l.inputs, delta_l);
        multy_cpu(delta_l, l.inputs, (float)2/l.group, 1);
    }
}
