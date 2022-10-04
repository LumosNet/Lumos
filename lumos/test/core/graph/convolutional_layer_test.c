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
    
}

int main()
{
    test_convolutional_layer_make();
    test_convolutional_layer_init();
    test_forward_convolutional_layer();
    return 0;
}