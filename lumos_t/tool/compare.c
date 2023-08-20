#include "compare.h"

int compare_array(void *a, void *b, char *type, int num)
{
    if (0 == strcmp(type, "int") || 0 == strcmp(type, "int g")){
        return compare_int_array((int*)a, (int*)b, num);
    } else if (0 == strcmp(type, "float") || 0 == strcmp(type, "float g")){
        return compare_float_array((float*)a, (float*)b, num);
    } else {
        fprintf(stderr, "Not Supported DataType!\n");
    }
    return 0;
}

int compare_float_array(float *a, float *b, int num)
{
    for (int i = 0; i < num; ++i){
        if (fabs(a[i]-b[i]) > 1e-5){
            printf("%f %f\n", a[i], b[i]);
            return ERROR;
        }
    }
    return PASS;
}

int compare_int_array(int *a, int *b, int num)
{
    for (int i = 0; i < num; ++i){
        if (fabs(a[i]-b[i]) > 1e-5){
            return ERROR;
        }
    }
    return PASS;
}

int compare_array_gpu(void *a, void *b, char *type, int num)
{
    if (0 == strcmp(type, "int")){
        return compare_int_array(a, b, num);
    } else if (0 == strcmp(type, "float")){
        return compare_float_array(a, b, num);
    } else if (0 == strcmp(type, "int g")){
        return compare_int_array_gpu(a, b, num);
    } else if (0 == strcmp(type, "float g")){
        return compare_float_array_gpu(a, b, num);
    } else {
        fprintf(stderr, "Not Supported DataType!\n");
    }
    return 0;
} 

int compare_float_array_gpu(float *a, float *b, int num)
{
    float *a_c = malloc(num*sizeof(float));
    float *b_c = malloc(num*sizeof(float));
    cudaMemcpy(a_c, a, num*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b_c, b, num*sizeof(float), cudaMemcpyDeviceToHost);
    return compare_float_array(a_c, b_c, num);
}

int compare_int_array_gpu(int *a, int *b, int num)
{
    int *a_c = malloc(num*sizeof(int));
    int *b_c = malloc(num*sizeof(int));
    cudaMemcpy(a_c, a, num*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b_c, b, num*sizeof(int), cudaMemcpyDeviceToHost);
    return compare_int_array(a_c, b_c, num);
}
