#include <stdio.h>
#include <stdlib.h>

#include "layer.h"
#include "convolutional_layer.h"
#include "utest.h"

void test_convolutional_layer_make()
{
    test_run("test_convolutional_layer_make");
    Layer *l;
    l = make_convolutional_layer(3, 3, 1, 0, 1, 0, "logistic");
    if (l->type != CONVOLUTIONAL){
        test_msg("convolutional layer type error");
        test_res(1, "test_convolutional_layer_make.test_type");
        return;
    }
    if (l->bias != 1){
        test_msg("convolutional layer bias set error");
        test_res(1, "test_convolutional_layer_make.test_bias");
    }
    if (l->filters != 3){
        test_msg("convolutional layer filters set error");
        test_res(1, "test_convolutional_layer_make.test_filters");
        return;
    }
    if (0 != strcmp(l->active_str, "logistic")){
        test_msg("convolutional layer active func set error");
        test_res(1, "test_convolutional_layer_make.test_active");
        return;
    }
    if (l->active == NULL){
        test_msg("load active func error");
        test_res(1, "");
        return;
    }
    if (l->gradient == NULL){
        test_msg("load gradient func error");
        test_res(1, "");
        return;
    }
    if (l->forward == NULL){
        test_msg("load forward func error");
        test_res(1, "");
        return;
    }
    if (l->backward == NULL){
        test_msg("load backward func error");
        test_res(1, "");
        return;
    }
    if (l->update == NULL){
        test_msg("load update func error");
        test_res(1, "");
        return;
    }
    if (l->init_layer_weights == NULL){
        test_msg("load init_layer_weights func error");
        test_res(1, "");
        return;
    }
    free(l);
    l = make_convolutional_layer(3, 3, 1, 0, 0, 0, "logistic");
    if (l->bias != 0){
        test_msg("convolutional set bias error");
        test_res(1, "");
    }
    test_res(0, "");
}


void test_convolutional_layer_init()
{
    test_run("test_convolutional_layer_init");
    Layer *l;
    l = make_convolutional_layer(3, 3, 1, 0, 1, 0, "logistic");
    init_convolutional_layer(l, 4, 4, 1);
    if (l->input_h != 4 || l->input_w != 4 || l->input_c != 1){
        test_res(1, "convolutional inputs error");
        return;
    }
    if (l->inputs != 16){
        test_res(1, "convolutional inputs error");
        return;
    }
    if (l->outputs != 4*l->filters || l->ksize != 3){
        test_res(1, "convolutional outputs error");
        return;
    }
    if (l->kernel_weights_size != l->filters*l->ksize*l->ksize*l->input_c){
        test_res(1, "convolutional kernel weights size error");
        return;
    }
    if (l->bias_weights_size != l->filters){
        test_res(1, "convolutional bias weights size error");
        return;
    }
    if (l->deltas != l->inputs){
        test_res(1, "convolutional deltas size error");
        return;
    }
    free(l);
    l = make_convolutional_layer(3, 3, 1, 0, 0, 0, "logistic");
    init_convolutional_layer(l, 4, 4, 1);
    if (l->bias_weights_size != 0){
        test_res(1, "convolutional bias weights size error");
        return;
    }
    test_res(0, "");
}


void test_forward_convolutional_layer()
{
    test_run("test_forward_convolutional_layer");
    Layer *l;
    l = make_convolutional_layer(3, 3, 1, 0, 1, 0, "logistic");
    init_convolutional_layer(l, 4, 4, 1);
    float *input = calloc(16, sizeof(float));
    for (int i = 0; i < 16; ++i){
        input[i] = i*0.1;
    }
    float *output = calloc(l->outputs, sizeof(float));
    float *kernel_weights = calloc(l->kernel_weights_size, sizeof(float));
    float *update_kernel_weights = calloc(l->kernel_weights_size, sizeof(float));
    float *bias_weights = calloc(l->bias_weights_size, sizeof(float));
    float *update_bias_weights = calloc(l->bias_weights_size, sizeof(float));
    float *workspace = calloc(l->workspace_size, sizeof(float));
    for (int i = 0; i < l->kernel_weights_size; ++i){
        kernel_weights[i] = i*0.1;
    }
    memcpy(update_kernel_weights, kernel_weights, l->kernel_weights_size*sizeof(float));
    for (int i = 0; i < l->bias_weights_size; ++i){
        bias_weights[i] = 0.01;
    }
    memcpy(update_bias_weights, bias_weights, l->bias_weights_size*sizeof(float));
    l->input = input;
    l->output = output;
    l->workspace = workspace;
    l->kernel_weights = kernel_weights;
    l->update_kernel_weights = update_kernel_weights;
    l->bias_weights = bias_weights;
    l->update_bias_weights = update_bias_weights;
    // 0.0  0.1  0.2  0.3
    // 0.4  0.5  0.6  0.7
    // 0.8  0.9  1.0  1.1
    // 1.2  1.3  1.4  1.5

    // 0.0  0.1  0.2    2.58    2.94
    // 0.3  0.4  0.5    4.02    4.38
    // 0.6  0.7  0.8

    // 0.9  1.0  1.1    6.63    7.8
    // 1.2  1.3  1.4    11.31   12.48
    // 1.5  1.6  1.7

    // 1.8  1.9  2.0    10.68   12.66
    // 2.1  2.2  2.3    18.6    20.58
    // 2.4  2.5  2.6

    // channel_1:      channel_2:      channel_3:
    // 2.59   2.95     6.64   7.81     10.69  12.67
    // 4.03   4.39     11.32  12.49    18.61  20.59

    // im2col:
    // 2.59 2.95 4.03 4.39 6.64 7.81 11.32 12.49 10.69 12.67 18.61 20.59
    float res[] = {2.59, 2.95, 4.03, 4.39, 6.64, 7.81, 11.32, 12.49, 10.69, 12.67, 18.61, 20.59};
    activate_list(res, 12, l->active);
    forward_convolutional_layer(*l, 1);
    for (int i = 0; i < l->outputs; ++i){
        if (fabs(l->output[i]-res[i]) > 1e-6){
            printf("index:%d  value:%f  res:%f\n", i, l->output[i], res[i]);
            test_res(1, "forward convolutional error");
            return;
        }
    }
    test_res(0, "");
}


void test_backward_convolutional_layer()
{
    test_run("test_backward_convolutional_layer");
    Layer *l;
    l = make_convolutional_layer(3, 3, 1, 0, 1, 0, "logistic");
    init_convolutional_layer(l, 4, 4, 1);
    float *output = calloc(l->outputs, sizeof(float));
    float *delta_n = calloc(l->outputs, sizeof(float));
    float *delta_l = calloc(l->inputs, sizeof(float));
    float *workspace = calloc(l->workspace_size, sizeof(float));
    for (int i = 0; i < l->outputs; ++i){
        output[i] = i*0.1;
    }
    for (int i = 0; i < l->outputs; ++i){
        delta_n[i] = i*0.1;
    }
    l->output = output;
    l->delta = delta_l;
    l->workspace = workspace;
    
}


void test_update_convolutional_layer()
{
    test_run("test_update_convolutional_layer");
    Layer *l;
    l = make_convolutional_layer(3, 3, 1, 0, 1, 0, "logistic");
    init_convolutional_layer(l, 4, 4, 1);
    float *input = calloc(l->inputs, sizeof(float));
    float *delta_n = calloc(l->outputs, sizeof(float));
    float *workspace = calloc(l->workspace_size, sizeof(float));
    float *update_kernel_weights = calloc(l->kernel_weights_size, sizeof(float));
    float *update_bias_weights = calloc(l->bias_weights_size, sizeof(float));
    for (int i = 0; i < l->inputs; ++i){
        input[i] = i*0.1;
    }
    for (int i = 0; i < l->outputs; ++i){
        delta_n[i] = i*0.1;
    }
    for (int i = 0; i < l->kernel_weights_size; ++i){
        update_kernel_weights[i] = i*0.1;
    }
    for (int i = 0; i < l->bias_weights_size; ++i){
        update_bias_weights[i] = i*0.1;
    }
    l->input = input;
    l->workspace = workspace;
    l->update_kernel_weights = update_kernel_weights;
    l->update_bias_weights = update_bias_weights;
    l->update(*l, 0.1, 1, delta_n);

    // input:
    // 0.0  0.1  0.2  0.3
    // 0.4  0.5  0.6  0.7
    // 0.8  0.9  1.0  1.1
    // 1.2  1.3  1.4  1.5

    // im2col:
    // 0.0  0.1  0.4  0.5
    // 0.1  0.2  0.5  0.6
    // 0.2  0.3  0.6  0.7
    // 0.4  0.5  0.8  0.9
    // 0.5  0.6  0.9  1.0
    // 0.6  0.7  1.0  1.1
    // 0.8  0.9  1.2  1.3
    // 0.9  1.0  1.3  1.4
    // 1.0  1.1  1.4  1.5

    // im2col_transpose:
    // 0.0  0.1  0.2  0.4  0.5  0.6  0.8  0.9  1.0
    // 0.1  0.2  0.3  0.5  0.6  0.7  0.9  1.0  1.1
    // 0.4  0.5  0.6  0.8  0.9  1.0  1.2  1.3  1.4
    // 0.5  0.6  0.7  0.9  1.0  1.1  1.3  1.4  1.5

    // delta_n:
    // 0.0  0.1  0.2  0.3
    // 0.4  0.5  0.6  0.7
    // 0.8  0.9  1.0  1.1

    // gemm:
    // 0.24  0.30  0.36  0.48  0.54  0.60  0.72  0.78  0.84
    // 0.64  0.86  1.08  1.52  1.74  1.96  2.40  2.62  2.84
    // 1.04  1.42  1.80  2.56  2.94  3.32  4.08  4.46  4.84

    // update_kernel_weights:
    // 0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8
    // 0.9  1.0  1.1  1.2  1.3  1.4  1.5  1.6  1.7
    // 1.8  1.9  2.0  2.1  2.2  2.3  2.4  2.5  2.6

    // update_res(rate=0.01):
    // 0.024  0.130  0.236  0.348  0.454  0.560  0.672  0.778  0.884
    // 0.964  1.086  1.208  1.352  1.474  1.596  1.740  1.862  1.984
    // 1.904  2.042  2.180  2.356  2.494  2.632  2.808  2.946  3.084

    float compare_kernel_weights[] = \
    {0.024,  0.130,  0.236,  0.348,  0.454,  0.560,  0.672,  0.778,  0.884, \
     0.964,  1.086,  1.208,  1.352,  1.474,  1.596,  1.740,  1.862,  1.984, \
     1.904,  2.042,  2.180,  2.356,  2.494,  2.632,  2.808,  2.946,  3.084};

    for (int i = 0; i < l->kernel_weights_size; ++i){
        if (fabs(l->update_kernel_weights[i]-compare_kernel_weights[i]) > 1e-6){
            printf("index:%d  value:%f  res:%f\n", i, l->update_kernel_weights[i], compare_kernel_weights[i]);
            test_res(1, "update convolutional error");
            return;
        }
    }
    test_res(0, "");
}


int main()
{
    test_convolutional_layer_make();
    test_convolutional_layer_init();
    test_forward_convolutional_layer();
    test_update_convolutional_layer();
    return 0;
}