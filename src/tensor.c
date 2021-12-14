#include "tensor.h"

Tensor *tensor_d(int dim, int *size)
{
    Tensor *ts = malloc(sizeof(Tensor));
    ts->size = malloc(dim*sizeof(int));
    memcpy(ts->size, size, dim*sizeof(int));
    ts->type = TENSOR;
    ts->dim = dim;
    ts->num = multing_int_list(size, 0, dim);
    ts->data = calloc(ts->num, sizeof(float));
    return ts;
}

Tensor *tensor_x(int dim, int *size, float x)
{
    Tensor *ts = tensor_d(dim, size);
    if (x != 0) full_list_with_float(ts->data, x, ts->num, 0, 0);
    return ts;
}

Tensor *tensor_list(int dim, int *size, float *list)
{
    Tensor *ts = tensor_d(dim, size);
    memcpy_float_list(ts->data, list, 0, 0, ts->num);
    return ts;
}

Tensor *tensor_sparse(int dim, int *size, int **index, float *list, int n)
{
    Tensor *ts = tensor_d(dim, size);
    for (int i = 0; i < n; ++i){
        int lindex = index_ts2ls(index[i], dim, size);
        ts->data[lindex] = list[i];
    }
    return ts;
}

Tensor *tensor_copy(Tensor *ts)
{
    Tensor *ret_ts = tensor_list(ts->dim, ts->size, ts->data);
    return ret_ts;
}

void resize(Tensor *ts, int dim, int *size)
{
    float *data = ts->data;
    int *t_size = ts->size;
    ts->dim = dim;
    ts->num = multing_int_list(size, 0, dim);
    ts->data = malloc(ts->num*sizeof(float));
    ts->size = malloc(ts->dim*sizeof(int));
    memcpy_float_list(ts->data, data, 0, 0, ts->num);
    memcpy_int_list(ts->size, size, 0, 0, dim);
    free(data);
    free(t_size);
}

void tsprint(Tensor *ts)
{
    printf("dimenssion : %d\n", ts->dim);
    printf("instruction : ");
    for (int i = 0; i < ts->dim-1; ++i){
        printf("%d * ", ts->size[i]);
    }
    printf("%d\n", ts->size[ts->dim-1]);
    printf("data num : %d\n", ts->num);
    if (ts->dim == 1){
        for (int i = 0; i < ts->num; ++i){
            printf("%f ", ts->data[i]);
        }
        printf("\n");
    }
    else{
        for (int i = 0; i < ts->num; ++i){
            if ((i+1) % ts->size[0] == 0) printf(" %f\n", ts->data[i]);
            else printf(" %f", ts->data[i]);
            if ((i+1) % (ts->size[0]*ts->size[1]) == 0) printf("\n");
        }
    }
}

float get_pixel(float *data, int dim, int *size, int *index)
{
    int num = multing_int_list(size, 0, dim);
    int ts2ls = index_ts2ls(index, dim, size);
    if (ts2ls >= 0 && ts2ls < num) return data[ts2ls];
    return 0;
}

void change_pixel(float *data, int dim, int *size, int *index, float x)
{
    int ts2ls = index_ts2ls(index, dim, size);
    if (ts2ls) data[ts2ls] = x;
}

float min(float *data, int num)
{
    float min = data[0];
    for (int i = 1; i < num; ++i){
        if (data[i] < min) min = data[i];
    }
    return min;
}

float max(float *data, int num)
{
    float max = data[0];
    for (int i = 1; i < num; ++i){
        if (data[i] > max) max = data[i];
    }
    return max;
}

float mean(float *data, int num)
{
    float sum = sum_float_list(data, 0, num);
    return sum / (float)num;
}

void add_x(float *data, int num, float x)
{
    for (int i = 0; i < num; ++i){
        data[i] += x;
    }
}

void mult_x(float *data, int num, float x)
{
    for (int i = 0; i < num; ++i){
        data[i] *= x;
    }
}

void add(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i){
        space[i] = data_a[i] + data_b[i];
    }
}

void subtract(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i){
        space[i] = data_a[i] - data_b[i];
    }
}

void multiply(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i){
        space[i] = data_a[i] * data_b[i];
    }
}

void divide(float *data_a, float *data_b, int num, float *space)
{
    for (int i = 0; i < num; ++i){
        space[i] = data_a[i] / data_b[i];
    }
}

void saxpy(float *data_a, float *data_b, int num, float x, float *space)
{
    for (int i = 0; i < num; ++i){
        space[i] = data_a[i] + data_b[i]*x;
    }
}