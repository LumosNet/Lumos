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
        if (fabs(data[i]-0.1) < 1e-6){
            test_res(0, "");
            return;
        }
    }
    test_res(1, "");
}

void test_fill_cpu_int()
{
    test_run("test_fill_cpu_int");
    float *data = calloc(10, sizeof(float));
    fill_cpu(data, 10, 2, 1);
    for (int i = 0; i < 10; ++i){
        if (fabs(data[i]-2) > 1e-6){
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

void test_multy_cpu()
{
    test_run("test_multy_cpu");
    float *data = calloc(10, sizeof(float));
    fill_cpu(data, 10, 1, 1);
    multy_cpu(data, 10, 2, 1);
    for (int i = 0; i < 10; ++i){
        if (data[i] != 2){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

void test_multy_cpu_offset()
{
    test_run("test_multy_cpu_offset");
    float *data = calloc(10, sizeof(float));
    fill_cpu(data, 10, 1, 1);
    multy_cpu(data, 10, 2, 2);
    for (int i = 0; i < 10; i+=2){
        if (data[i] != 2){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

void test_add_cpu()
{
    test_run("test_add_cpu");
    float *data = calloc(10, sizeof(float));
    add_cpu(data, 10, 2, 1);
    for (int i = 0; i < 10; ++i){
        if (data[i] != 2){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

void test_add_cpu_offset()
{
    test_run("test_add_cpu_offset");
    float *data = calloc(10, sizeof(float));
    add_cpu(data, 10, 2, 2);
    for (int i = 0; i < 10; i+=2){
        if (data[i] != 2){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

void test_min_cpu()
{
    test_run("test_min_cpu");
    float *data = calloc(10, sizeof(float));
    for (int i = 0; i < 10; ++i){
        data[i] = i + 1;
    }
    float x = min_cpu(data, 10);
    if (fabs(x-1) > 1e-6){
        test_res(1, "");
        return;
    }
    test_res(0, "");
}

void test_max_cpu()
{
    test_run("test_max_cpu");
    float *data = calloc(10, sizeof(float));
    for (int i = 0; i < 10; ++i){
        data[i] = i + 1;
    }
    float x = max_cpu(data, 10);
    if (fabs(x-10) > 1e-6){
        test_res(1, "");
        return;
    }
    test_res(0, "");
}

void test_sum_cpu()
{
    test_run("test_sum_cpu");
    float *data = calloc(10, sizeof(float));
    int y = 0;
    for (int i = 0; i < 10; ++i){
        data[i] = i + 1;
        y += data[i];
    }
    float x = sum_cpu(data, 10);
    if (fabs(x-y) > 1e-6){
        test_res(1, "");
        return;
    }
    test_res(0, "");
}

void test_mean_cpu()
{
    test_run("test_mean_cpu");
    float *data = calloc(10, sizeof(float));
    float y = 0;
    for (int i = 0; i < 10; ++i){
        data[i] = i + 1;
        y += data[i];
    }
    y = y / (float)10;
    float x = mean_cpu(data, 10);
    if (fabs(x-y) > 1e-6){
        test_res(1, "");
        return;
    }
    test_res(0, "");
}

void test_one_hot_encoding()
{
    test_run("test_one_hot_encoding");
    float *data = calloc(10, sizeof(float));
    int flag = 1;
    for (int i = 0; i < 10; ++i){
        one_hot_encoding(10, i, data);
        for (int j = 0; j < 10; ++j){
            if (i == j && fabs(data[j]-1) > 1e-6) flag = 0;
            else if(i != j && fabs(data[j]-0) > 1e-6) flag = 0;
        }
        if (flag == 0){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

void test_add()
{
    test_run("test_add");
    float *a = calloc(10, sizeof(float));
    float *b = calloc(10, sizeof(float));
    float *res = calloc(10, sizeof(float));
    int flag = 0;
    for (int i = 0; i < 10; ++i){
        a[i] = i;
        b[i] = i;
    }
    add(a, b, 10, res);
    for (int i = 0; i < 10; ++i){
        if (fabs(res[i]-i-i) > 1e-6) flag = 0;
    }
    if (flag == 0){
        test_res(0, "");
        return;
    }
    for (int i = 0; i < 10; ++i){
        a[i] = i;
        b[i] = -i;
    }
    add(a, b, 10, res);
    for (int i = 0; i < 10; ++i){
        if (fabs(res[i]) > 1e-6) flag = 0;
    }
    if (flag == 0){
        test_res(0, "");
        return;
    }
    test_res(1, "");
}

void test_subtract()
{
    test_run("test_subtract");
    float *a = calloc(10, sizeof(float));
    float *b = calloc(10, sizeof(float));
    float *res = calloc(10, sizeof(float));
    int flag = 0;
    for (int i = 0; i < 10; ++i){
        a[i] = i;
        b[i] = i;
    }
    subtract(a, b, 10, res);
    for (int i = 0; i < 10; ++i){
        if (fabs(res[i]) > 1e-6) flag = 0;
    }
    if (flag == 0){
        test_res(0, "");
        return;
    }
    test_res(1, "");
}

void test_multiply()
{
    test_run("test_multiply");
    float *a = calloc(10, sizeof(float));
    float *b = calloc(10, sizeof(float));
    float *res = calloc(10, sizeof(float));
    int flag = 0;
    for (int i = 0; i < 10; ++i){
        a[i] = i;
        b[i] = i;
    }
    multiply(a, b, 10, res);
    for (int i = 0; i < 10; ++i){
        if (fabs(res[i]-i*i) > 1e-6) flag = 0;
    }
    if (flag == 0){
        test_res(0, "");
        return;
    }
    test_res(1, "");
}

void test_divide()
{
    test_run("test_divide");
    float *a = calloc(10, sizeof(float));
    float *b = calloc(10, sizeof(float));
    float *res = calloc(10, sizeof(float));
    int flag = 0;
    for (int i = 0; i < 10; ++i){
        a[i] = i;
        b[i] = i;
    }
    divide(a, b, 10, res);
    for (int i = 0; i < 10; ++i){
        if (fabs(res[i]-1) > 1e-6) flag = 0;
    }
    if (flag == 0){
        test_res(0, "");
        return;
    }
    test_res(1, "");
}

void test_saxpy()
{
    test_run("test_saxpy");
    float *a = calloc(10, sizeof(float));
    float *b = calloc(10, sizeof(float));
    float *res = calloc(10, sizeof(float));
    int flag = 0;
    for (int i = 0; i < 10; ++i){
        a[i] = i;
        b[i] = i;
    }
    saxpy(a, b, 10, -2, res);
    for (int i = 0; i < 10; ++i){
        if (fabs(res[i]+i) > 1e-6) flag = 0;
    }
    if (flag == 0){
        test_res(0, "");
        return;
    }
    test_res(1, "");
}

int main()
{
    test_fill_cpu_float();
    test_fill_cpu_int();
    test_fill_cpu_offset();
    test_multy_cpu();
    test_multy_cpu_offset();
    test_add_cpu();
    test_add_cpu_offset();
    test_min_cpu();
    test_max_cpu();
    test_sum_cpu();
    test_mean_cpu();
    test_one_hot_encoding();
    test_add();
    test_subtract();
    test_multiply();
    test_divide();
    test_saxpy();
}