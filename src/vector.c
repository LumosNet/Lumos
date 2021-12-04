#include "vector.h"

Tensor *Tensor_x(int num, int flag, float x)
{
    if (flag) return array_x(num, 1, x);
    else return array_x(1, num, x);
}

Tensor *Tensor_list(int num, int flag, float *list)
{
    if (flag) return array_list(num, 1, list);
    else return array_list(1, num, list);
}

/* 0代表行向量, 1代表列向量 */
int colorrow(Tensor *v)
{
    if (v->size[0] > 1) return 0;
    return 1;
}

float ts_get_pixel_vt(Tensor *v, int index)
{
    return v->data[index];
}

void ts_change_pixel_vt(Tensor *v, int index, float x)
{
    v->data[index] = x;
}

void replace_vtlist(Tensor *v, float *list)
{
    for (int i = 0; i < v->num; ++i){
        v->data[i] = list[i];
    }
}

void replace_vtx(Tensor *v, float x)
{
    for (int i = 0; i < v->num; ++i){
        v->data[i] = x;
    }
}

void del_pixel(Tensor *v, int index)
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

void insert_pixel(Tensor *v, int index, float x)
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

Tensor *merge_vt(Tensor *a, Tensor *b, int index)
{
    int flag = a->size[0] > a->size[1] ? 0 : 1;
    Tensor *res = Tensor_x(a->num + b->num, flag, 0);
    memcpy(res->data, a->data, a->num*sizeof(float));
    memcpy(res->data+a->num, b->data, b->num*sizeof(float));
    return res;
}

Tensor *slice_vt(Tensor *v, int index_h, int index_t)
{
    int flag = v->size[0] > v->size[1] ? 0 : 1;
    Tensor *res = Tensor_x(index_t-index_h, flag, 0);
    memcpy(res->data, v->data+index_h, (index_t-index_h)*sizeof(float));
    return res;
}

Tensor *add_vt(Tensor *a, Tensor *b)
{
    Tensor *res = tensor_copy(a);
    for (int i = 0; i < a->num; ++i){
        res->data[i] += b->data[i];
    }
    return res;
}

Tensor *subtract_vt(Tensor *a, Tensor *b)
{
    Tensor *res = tensor_copy(a);
    for (int i = 0; i < a->num; ++i){
        res->data[i] -= b->data[i];
    }
    return res;
}

Tensor *divide_vt(Tensor *a, Tensor *b)
{
    Tensor *res = tensor_copy(a);
    for (int i = 0; i < a->num; ++i){
        res->data[i] /= (float)b->data[i];
    }
    return res;
}

Tensor *multiply_vt(Tensor *a, Tensor *b)
{
    Tensor *res = tensor_copy(a);
    for (int i = 0; i < a->num; ++i){
        res->data[i] *= b->data[i];
    }
    return res;
}

void add_vtx(Tensor *v, float x)
{
    for (int i = 0; i < v->num; ++i){
        v->data[i] += x;
    }
}

void Tensor_multx(Tensor *v, float x)
{
    for (int i = 0; i < v->num; ++i){
        v->data[i] *= x;
    }
}

float norm1_vt(Tensor *v)
{
    float res = 0;
    for (int i = 0; i < v->num; ++i){
        res += fabs(v->data[i]);
    }
    return res;
}

float norm2_vt(Tensor *v)
{
    float res = 0;
    for (int i = 0; i < v->num; ++i){
        res += v->data[i] * v->data[i];
    }
    res = sqrt(res);
    return res;
}

float normp_vt(Tensor *v, int p)
{
    float res = 0;
    for (int i = 0; i < v->num; ++i){
        res += pow(v->data[i], (double)p);
    }
    res = pow(res, (double)1/p);
    return res;
}

float infnorm_vt(Tensor *v)
{
    float res = ts_max(v);
    return res;
}

float ninfnorm_vt(Tensor *v)
{
    float res = ts_min(v);
    return res;
}