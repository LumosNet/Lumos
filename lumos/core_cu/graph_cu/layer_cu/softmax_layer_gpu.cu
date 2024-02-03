#include "softmax_layer_gpu.h"

void init_softmax_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h*l->input_w*l->input_c;

    l->output_h = h;
    l->output_w = w;
    l->output_c = c;
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->workspace_size = l->inputs+1;

    cudaMalloc((void**)&l->output, l->outputs*subdivision*sizeof(float));
    cudaMalloc((void**)&l->delta, l->outputs*subdivision*sizeof(float));

    fprintf(stderr, "Softmax         Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_softmax_layer_gpu(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        softmax_exp_sum_gpu(input, l.inputs, l.workspace, l.workspace+l.inputs);
        softmax_gpu(l.workspace, l.inputs, output, l.workspace+l.inputs);
    }
}

void backward_softmax_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        float *input = l.input + offset_i;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        softmax_exp_sum_gpu(input, l.inputs, l.workspace, l.workspace+l.inputs);
        softmax_grident_gpu(l.workspace, l.inputs, delta_l, l.workspace+l.inputs);
        matrix_multiply_gpu(delta_n, delta_l, l.inputs, delta_l);
    }
}