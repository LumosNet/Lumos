#include "compare.h"

int compare_float_array(float *a, float *b, int num)
{
    for (int i = 0; i < num; ++i){
        if (fabs(a[i]-b[i]) > 1e-6){
            return ERROR;
        }
    }
    return PASS;
}

int compare_int_array(int *a, int *b, int num)
{
    for (int i = 0; i < num; ++i){
        if (fabs(a[i]-b[i]) > 1e-6){
            return ERROR;
        }
    }
    return PASS;
}

int compare_string_array(int *a, int *b, int num)
{
    for (int i = 0; i < num; ++i){
        if (0 != strcmp(a[i], b[i])){
            return ERROR;
        }
    }
    return PASS;
}
