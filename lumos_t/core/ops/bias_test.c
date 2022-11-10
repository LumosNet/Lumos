#include <stdio.h>
#include <stdlib.h>

#include "bias.h"
#include "utest.h"

void test_add_bias_list()
{
    test_run("test_add_bias_list");
    float *origin = malloc(5*sizeof(float));
    float *bias = malloc(5*sizeof(float));
    for (int i = 0; i < 5; ++i){
        origin[i] = i;
        bias[i] = i+1;
    }
    add_bias(origin, bias, 5, 1);
    for (int i = 0; i < 5; ++i){
        if (origin[i] != 2*i+1){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

void test_add_bias_channel()
{
    test_run("test_add_bias_channel");
    float *origin = malloc(12*sizeof(float));
    float *bias = malloc(3*sizeof(float));
    for (int i = 0; i < 12; ++i){
        origin[i] = i;
    }
    for (int i = 0; i < 3; ++i){
        bias[i] = i+1;
    }
    add_bias(origin, bias, 3, 4);
    for (int i = 0; i < 4; ++i){
        if (origin[i] != i+1){
            test_res(1, "");
            return;
        }
    }
    for (int i = 0; i < 4; ++i){
        if (origin[4+i] != i+6){
            test_res(1, "");
            return;
        }
    }
    for (int i = 0; i < 4; ++i){
        if (origin[8+i] != i+11){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

int main()
{
    test_add_bias_list();
    test_add_bias_channel();
}