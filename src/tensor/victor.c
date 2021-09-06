#include "victor.h"

Victor *victor_x(int num, int flag, float x)
{
    if (flag) return array_x(num, 1, x);
    else return array_x(1, num, x);
}

Victor *victor_list(int num, int flag, float *list)
{
    if (flag) return array_list(num, 1, list);
    else return array_list(1, num, list);
}

/* 0代表行向量, 1代表列向量 */
int colorrow(Victor *v)
{
    if (v->size[0] > 1) return 0;
    return 1;
}

float get_pixel_vt(Victor *v, int index)
{
    return v->data[index];
}

void change_pixel_vt(Victor *v, int index, float x)
{
    v->data[index] = x;
}

void replace_vtlist(Victor *v, float *list)
{
    for (int i = 0; i < v->num; ++i){
        v->data[i] = list[i];
    }
}

void replace_vtx(Victor *v, float x)
{
    for (int i = 0; i < v->num; ++i){
        v->data[i] = x;
    }
}

void del_pixel(Victor *v, int index)
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

void insert_pixel(Victor *v, int index, float x)
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

Victor *merge_vt(Victor *a, Victor *b, int index)
{
    int flag = a->size[0] > a->size[1] ? 0 : 1;
    Victor *res = victor_x(a->num + b->num, flag, 0);
    memcpy(res->data, a->data, a->num*sizeof(float));
    memcpy(res->data+a->num, b->data, b->num*sizeof(float));
    return res;
}

Victor *slice_vt(Victor *v, int index_h, int index_t)
{
    int flag = v->size[0] > v->size[1] ? 0 : 1;
    Victor *res = victor_x(index_t-index_h, flag, 0);
    memcpy(res->data, v->data+index_h, (index_t-index_h)*sizeof(float));
    return res;
}

Victor *add_vt(Victor *a, Victor *b)
{
    Victor *res = copy(a);
    for (int i = 0; i < a->num; ++i){
        res->data[i] += b->data[i];
    }
    return res;
}

Victor *subtract_vt(Victor *a, Victor *b)
{
    Victor *res = copy(a);
    for (int i = 0; i < a->num; ++i){
        res->data[i] -= b->data[i];
    }
    return res;
}

Victor *divide_vt(Victor *a, Victor *b)
{
    Victor *res = copy(a);
    for (int i = 0; i < a->num; ++i){
        res->data[i] /= (float)b->data[i];
    }
    return res;
}

Victor *multiply_vt(Victor *a, Victor *b)
{
    Victor *res = copy(a);
    for (int i = 0; i < a->num; ++i){
        res->data[i] *= b->data[i];
    }
    return res;
}

void add_vtx(Victor *v, float x)
{
    for (int i = 0; i < v->num; ++i){
        v->data[i] += x;
    }
}

void victor_multx(Victor *v, float x)
{
    for (int i = 0; i < v->num; ++i){
        v->data[i] *= x;
    }
}

float norm1_vt(Victor *v)
{
    float res = 0;
    for (int i = 0; i < v->num; ++i){
        res += fabs(v->data[i]);
    }
    return res;
}

float norm2_vt(Victor *v)
{
    float res = 0;
    for (int i = 0; i < v->num; ++i){
        res += v->data[i] * v->data[i];
    }
    res = sqrt(res);
    return res;
}

float normp_vt(Victor *v, int p)
{
    float res = 0;
    for (int i = 0; i < v->num; ++i){
        res += pow(v->data[i], (double)p);
    }
    res = pow(res, (double)1/p);
    return res;
}

float infnorm_vt(Victor *v)
{
    float res = get_max(v);
    return res;
}

float ninfnorm_vt(Victor *v)
{
    float res = get_min(v);
    return res;
}