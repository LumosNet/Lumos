#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "gemm.h"
#include "utest.h"

void test_gemm_nn()
{
    /*
        1 2 3  1 2 3
        4 5 6  4 5 6
        7 8 9  7 8 9

        30  36  42
        66  81  96
        102 126 150
    */
    test_run("test_gemm_nn");
    float *a = calloc(9, sizeof(float));
    float *b = calloc(9, sizeof(float));
    float *c = calloc(9, sizeof(float));
    float *x = calloc(9, sizeof(float));
    for (int i = 0; i < 9; ++i){
        a[i] = i+1;
        b[i] = i+1;
    }
    x[0] = 30;
    x[1] = 36;
    x[2] = 42;
    x[3] = 66;
    x[4] = 81;
    x[5] = 96;
    x[6] = 102;
    x[7] = 126;
    x[8] = 150;
    gemm_nn(3, 3, 3, 3, 1, a, b, c);
    for (int i = 0; i < 9; ++i){
        if (fabs(c[i]-x[i]) > 1e-6){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

void test_gemm_tn()
{
    /*
        1 2 3  1 2 3
        4 5 6  4 5 6
        7 8 9  7 8 9

        66  78  90
        78  93  108
        90 108  126
    */
    test_run("test_gemm_tn");
    float *a = calloc(9, sizeof(float));
    float *b = calloc(9, sizeof(float));
    float *c = calloc(9, sizeof(float));
    float *x = calloc(9, sizeof(float));
    for (int i = 0; i < 9; ++i){
        a[i] = i+1;
        b[i] = i+1;
    }
    x[0] = 66;
    x[1] = 78;
    x[2] = 90;
    x[3] = 78;
    x[4] = 93;
    x[5] = 108;
    x[6] = 90;
    x[7] = 108;
    x[8] = 126;
    gemm_tn(3, 3, 3, 3, 1, a, b, c);
    for (int i = 0; i < 9; ++i){
        if (fabs(c[i]-x[i]) > 1e-6){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

void test_gemm_nt()
{
    /*
        1 2 3  1 2 3
        4 5 6  4 5 6
        7 8 9  7 8 9

        14  32   50
        32  77   122
        50  122  194
    */
    test_run("test_gemm_nt");
    float *a = calloc(9, sizeof(float));
    float *b = calloc(9, sizeof(float));
    float *c = calloc(9, sizeof(float));
    float *x = calloc(9, sizeof(float));
    for (int i = 0; i < 9; ++i){
        a[i] = i+1;
        b[i] = i+1;
    }
    x[0] = 14;
    x[1] = 32;
    x[2] = 50;
    x[3] = 32;
    x[4] = 77;
    x[5] = 122;
    x[6] = 50;
    x[7] = 122;
    x[8] = 194;
    gemm_nt(3, 3, 3, 3, 1, a, b, c);
    for (int i = 0; i < 9; ++i){
        if (fabs(c[i]-x[i]) > 1e-6){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

void test_gemm_tt()
{
    /*
        1 2 3  1 2 3
        4 5 6  4 5 6
        7 8 9  7 8 9

        30  66  102
        36  81  126
        42  96  150
    */
    test_run("test_gemm_tt");
    float *a = calloc(9, sizeof(float));
    float *b = calloc(9, sizeof(float));
    float *c = calloc(9, sizeof(float));
    float *x = calloc(9, sizeof(float));
    for (int i = 0; i < 9; ++i){
        a[i] = i+1;
        b[i] = i+1;
    }
    x[0] = 30;
    x[1] = 66;
    x[2] = 102;
    x[3] = 36;
    x[4] = 81;
    x[5] = 126;
    x[6] = 42;
    x[7] = 96;
    x[8] = 150;
    gemm_tt(3, 3, 3, 3, 1, a, b, c);
    for (int i = 0; i < 9; ++i){
        if (fabs(c[i]-x[i]) > 1e-6){
            printf("%d %f %f\n", i, c[i], x[i]);
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

int main()
{
    test_gemm_nn();
    test_gemm_tn();
    test_gemm_nt();
    test_gemm_tt();
}