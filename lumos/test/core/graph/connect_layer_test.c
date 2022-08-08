#include <stdio.h>
#include <stdlib.h>

#include "layer.h"
#include "connect_layer.h"
#include "utest.h"

void test_connect_layer_make()
{
    test_run("test_connect_layer_make");
    Layer *l;
    l = make_connect_layer(4, 1, "relu");
    if (l->type != CONNECT){
        test_msg("connect layer type error");
        test_res(1, "test_connect_layer_make.test_type");
        return;
    }
    if (l->bias != 1){
        test_msg("connect layer bias set error");
        test_res(1, "test_connect_layer_make.test_bias");
        return;
    }
    if (l->filters != 1){
        test_msg("connect layer filters set error");
        test_res(1, "test_connect_layer_make.test_filters");
        return;
    }
    if (l->weights != 1){
        test_msg("connect layer weights set error");
        test_res(1, "test_connect_layer_make.test_weights");
        return;
    }
    if (0 != strcmp(l->active_str, "relu")){
        test_msg("connect layer active func set error");
        test_res(1, "test_connect_layer_make.test_active");
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
    l = make_connect_layer(4, 0, "relu");
    if (l->bias != 0){
        test_msg("connect set bias error");
        test_res(1, "");
    }
    test_res(0, "");
}

void test_connect_layer_init()
{
    test_run("test_connect_layer_init");
    Layer *l;
    l = make_connect_layer(4, 1, "relu");
    init_connect_layer(l, 1, 2, 1);
    if (l->input_h != 2 || l->input_w != 1 || l->input_c != 1){
        test_res(1, "connect inputs error");
        return;
    }
    if (l->inputs != 2){
        test_res(1, "connect inputs error");
        return;
    }
    if (l->outputs != 4 || l->ksize != 4){
        test_res(1, "connect outputs error");
        return;
    }
    if (l->workspace_size != 8){
        test_res(1, "connect workspace size error");
        return;
    }
    if (l->kernel_weights_size != 8){
        test_res(1, "connect kernel weights size error");
        return;
    }
    if (l->bias_weights_size != 4){
        test_res(1, "connect bias weights size error");
        return;
    }
    if (l->deltas != 2){
        test_res(1, "connect deltas size error");
        return;
    }
    l = make_connect_layer(4, 0, "relu");
    init_connect_layer(l, 1, 2, 1);
    if (l->bias_weights_size != 0){
        test_res(1, "connect bias weights size error");
        return;
    }
    test_res(0, "");
}

void test_forward_connect_layer()
{
    Layer *l;
    l = make_connect_layer(4, 1, "relu");
    init_connect_layer(l, 1, 2, 1);
    float *input = malloc(2*sizeof(float));
    input[0] = 1;   // 1
    input[1] = 2;   // 2
    float *output = calloc(4, sizeof(float));
    float *kernel_weights = calloc(8, sizeof(float));
    float *update_kernel_weights = calloc(8, sizeof(float));
    float *bias_weights = calloc(4, sizeof(float));
    float *update_bias_weights = calloc(4, sizeof(float));
    float *workspace = calloc(l->workspace_size, sizeof(float));
    kernel_weights[0] = 0.1;    // 0.1  0.2  1
    kernel_weights[1] = 0.2;    // 0.3  0.4  2
    kernel_weights[2] = 0.3;    // 0.5  0.6
    kernel_weights[3] = 0.4;    // 0.7  0.8
    kernel_weights[4] = 0.5;
    kernel_weights[5] = 0.6;
    kernel_weights[6] = 0.7;
    kernel_weights[7] = 0.8;
    memcpy(update_kernel_weights, kernel_weights, 8*sizeof(float));
    bias_weights[0] = 0.01;
    bias_weights[1] = 0.01;
    bias_weights[2] = 0.01;
    bias_weights[3] = 0.01;
    memcpy(update_bias_weights, bias_weights, 4*sizeof(float));
    l->input = input;
    l->output = output;
    l->workspace = workspace;
    l->kernel_weights = kernel_weights;
    l->update_kernel_weights = update_kernel_weights;
    l->bias_weights = bias_weights;
    l->update_bias_weights = update_bias_weights;
    /*
        0.5  0.51
        1.1  1.11
        1.7  1.71
        2.3  2.31
    */
    forward_connect_layer(*l, 1);
    if (fabs(output[0]-0.51) > 1e-6){
        test_res(1, "");
        return;
    }
    if (fabs(output[1]-1.11) > 1e-6){
        test_res(1, "");
        return;
    }
    if (fabs(output[2]-1.71) > 1e-6){
        test_res(1, "");
        return;
    }
    if (fabs(output[3]-2.31) > 1e-6){
        test_res(1, "");
        return;
    }
    test_res(0, "");
}

int main()
{
    test_connect_layer_make();
    test_connect_layer_init();
    test_forward_connect_layer();
}
