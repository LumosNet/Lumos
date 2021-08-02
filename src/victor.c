#include "victor.h"

Victor *create_Victor(int num, int flag)
{
    if (flag) return create_array(num, 1);
    else return create_array(1, num);
}

Victor *create_zeros_Victor(int num, int flag)
{
    if (flag) return create_zeros_array(num, 1);
    else return create_zeros_array(1, num);
}

Victor *create_Victor_full_x(int num, int flag, float x)
{
    if (flag) return create_array_full_x(num, 1, x);
    else return create_array_full_x(1, num, x);
}

Victor *list_to_Victor(int num, int flag, float *list)
{
    if (flag) return list_to_array(num, 1, list);
    else return list_to_array(1, num, list);
}

Victor *copy_victor(Victor *v)
{
    return copy_array(v);
}

float get_pixel_in_Victor(Victor *v, int index)
{
    return v->data[index];
}

float get_Victor_min(Victor *v)
{
    return get_array_min(v);
}

float get_Victor_max(Victor *v)
{
    return get_array_max(v);
}

float get_Victor_mean(Victor *v)
{
    return get_array_mean(v);
}

int pixel_num_Victor(Victor *v, float x)
{
    return pixel_num_array(v, x);
}

void change_pixel_in_Victor(Victor *v, int index, float x)
{
    v->data[index] = x;
}

void replace_Victor2list(Victor *v, float *list)
{
    for (int i = 0; i < v->num; ++i){
        v->data[i] = list[i];
    }
}

void replace_Victor_with_x(Victor *v, float x)
{
    for (int i = 0; i < v->num; ++i){
        v->data[i] = x;
    }
}

void del_pixel_in_Victor(Victor *v, int index)
{
    int flag = v->size[0] > v->size[1] ? 0 : 1;
    v->size[flag] -= 1;
    v->num -= 1;
    float *data = malloc(v->num * sizeof(float));
    memcpy(data, v->data, index*sizeof(float));
    memcpy(data+index, v->data+index+1, (v->num-index-1)*sizeof(float));
    free(v->data);
    v->data = data;
}

void insert_pixel_in_Victor(Victor *v, int index, float x)
{
    int flag = v->size[0] > v->size[1] ? 0 : 1;
    v->size[flag] += 1;
    v->num += 1;
    float *data = malloc(v->num * sizeof(float));
    memcpy(data, v->data, index*sizeof(float));
    memcpy(data+index+1, v->data+index, (v->num-index)*sizeof(float));
    data[index] = x;
    free(v->data);
    v->data = data;
}

Victor *merge_Victor(Victor *a, Victor *b, int index)
{
    int flag = a->size[0] > a->size[1] ? 0 : 1;
    Victor *res = create_Victor(a->num + b->num, flag);
    memcpy(res->data, a->data, a->num*sizeof(float));
    memcpy(res->data+a->num, b->data, b->num*sizeof(float));
    return res;
}

Victor *slice_Victor(Victor *v, int index_h, int index_t)
{
    int flag = v->size[0] > v->size[1] ? 0 : 1;
    Victor *res = create_Victor(index_t-index_h, flag);
    memcpy(res->data, v->data+index_h, (index_t-index_h)*sizeof(float));
    return res;
}

void del_Victor(Victor *v)
{
    del_array(v);
}

void exchange2pixel_in_Victor(Victor *v, int index_1, int index_2)
{
    float x = v->data[index_1];
    v->data[index_1] = v->data[index_2];
    v->data[index_2] = x;
}

float sum_Victor(Victor *v)
{
    return sum_array(v);
}

Victor *Victor_add(Victor *a, Victor *b)
{
    Victor *res = copy_victor(a);
    for (int i = 0; i < a->num; ++i){
        res->data[i] += b->data[i];
    }
    return res;
}

Victor *Victor_subtract(Victor *a, Victor *b)
{
    Victor *res = copy_victor(a);
    for (int i = 0; i < a->num; ++i){
        res->data[i] -= b->data[i];
    }
    return res;
}

Victor *Victor_divide(Victor *a, Victor *b)
{
    Victor *res = copy_victor(a);
    for (int i = 0; i < a->num; ++i){
        res->data[i] /= (float)b->data[i];
    }
    return res;
}

Victor *Victor_x_multiplication(Victor *a, Victor *b)
{
    Victor *res = copy_victor(a);
    for (int i = 0; i < a->num; ++i){
        res->data[i] *= b->data[i];
    }
    return res;
}

void Victor_add_x(Victor *v, float x)
{
    for (int i = 0; i < v->num; ++i){
        v->data[i] += x;
    }
}

void Victor_subtract_x(Victor *v, float x)
{
    for (int i = 0; i < v->num; ++i){
        v->data[i] -= x;
    }
}

void Victor_multiplication_x(Victor *v, float x)
{
    for (int i = 0; i < v->num; ++i){
        v->data[i] *= x;
    }
}

void Victor_divide_x(Victor *v, float x)
{
    for (int i = 0; i < v->num; ++i){
        v->data[i] /= (float)x;
    }
}

void Victor_saxpy(Victor *vx, Victor *vy, float x)
{
    array_saxpy(vx, vy, x);
}