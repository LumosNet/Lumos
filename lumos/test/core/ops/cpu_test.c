#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cpu.h"
#include "utest.h"

void test_fill_cpu_float()
{
    test_run("test_fill_cpu_float");
    float *data = calloc(10, sizeof(float));
    fill_cpu(data, 10, 0.1, 1);
    for (int i = 0; i < 10; ++i){
        if (fabs(data[i]-0.1) > 1e-6){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

void test_fill_cpu_int()
{
    test_run("test_fill_cpu_int");
    float *data = calloc(10, sizeof(float));
    fill_cpu(data, 10, 2, 1);
    for (int i = 0; i < 10; ++i){
        if (data[i] != 2){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

void test_fill_cpu_offset()
{
    test_run("test_fill_cpu_offset");
    float *data = calloc(10, sizeof(float));
    fill_cpu(data, 10, 1, 2);
    for (int i = 0; i < 10; i+=2){
        if (data[i] != 1){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

int main()
{
    test_fill_cpu_float();
    test_fill_cpu_int();
    test_fill_cpu_offset();
}