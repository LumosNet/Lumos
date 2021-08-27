#include "algebraic_space.h"

tensor *create(int dim, int *size, float x)
{
    tensor *ret = malloc(sizeof(tensor));
    int *m_size = malloc(dim*sizeof(int));
    memcpy(m_size, size, dim*sizeof(int));
    ret->dim = dim;
    ret->size = m_size;
    ret->num = multing_int_list(size, 0, dim);
    ret->data = calloc(ret->num, sizeof(float));
    if (x != 0) full_list_with_float(ret->data, x, ret->num, 0, 0);
    return ret;
}

tensor *list_to_tensor(int dim, int *size, float *list)
{
    tensor *ret = create(dim, size, 0);
    memcpy_float_list(ret->data, list, 0, 0, ret->num);
    return ret;
}

tensor *copy(tensor *m)
{
    tensor *ret = list_to_tensor(ts->dim, ts->size, ts->data);
    return ret;
}

int get_lindex(tensor *m, int *index)
{
    int ret = 0;
    for (int i = 0; i < ts->dim; ++i){
        int x = index[i]-1;
        for (int j = 0; j < i; ++j){
            x *= ts->size[j];
        }
        ret += x;
    }
    return ret;
}

int *get_mindex(tensor *m, int index)
{
    int *ret = malloc(ts->dim*sizeof(int));
    for (int i = ts->dim-1; i >= 0; --i){
        int x = 1;
        for (int j = 0; j < i; ++j){
            x *= ts->size[j];
        }
        ret[i] = (int)(index / x) + 1;
        index %= x;
    }
    return ret;
}

float get_pixel(tensor *m, int *index)
{
    int lindex = get_lindex(m, index);
    return get_float_in_list(ts->data, lindex);
}

void change_pixel(tensor *m, int *index, float x)
{
    int lindex = get_lindex(m, index);
    change_float_in_list(ts->data, x, lindex);
}

void replace_part(tensor *m, tensor *n, int *index)
{
    int *nindex = malloc(n->dim*sizeof(int));
    full_list_with_int(nindex, 1, n->dim, 1, 0);
    int *mindex = malloc(ts->dim*sizeof(int));
    full_list_with_int(mindex, 1, n->dim, 1, 0);
    while (__ergodic(n, nindex)){
        for (int i = 0; i < ts->dim; ++i){
            mindex[i] = nindex[i] + index[i] - 1;
        }
        change_pixel(m, mindex, get_pixel(n, nindex));
    }
}

int __ergodic(tensor *m, int *index)
{
    int res = 1;
    int dim = ts->dim - 1;
    while (index[dim] == ts->size[dim]){
        index[dim] = 1;
        dim -= 1;
        if (dim == -1){
            res = 0;
            break;
        }
    }
    if (res) index[dim] += 1;
    return res;
}

void resize(tensor *m, int dim, int *size)
{
    float *data = ts->data;
    ts->data = malloc(ts->num*sizeof(float));
    ts->dim = dim;
    memcpy_float_list(ts->data, data, 0, 0, multing_int_list(size, 0, dim));
    memcpy_void_list(ts->size, size, INT, 0, 0, ts->dim);
    free(data);
}

void slice(tensor *m, float *workspace, int *dim_c, int **size_c)
{
    int **sections = malloc(ts->dim*sizeof(int*));
    for (int i = 0; i < dim_c[0]; ++i){
        int *section = malloc(3*sizeof(int));
        section[0] = 1;
        section[1] = 0;
        section[2] = ts->size[i];
        sections[i] = section;
    }
    int dim_c_index = 0;
    for (int i = dim_c[0]; i < ts->dim; ++i){
        if (i == dim_c[dim_c_index]){
            sections[i] = size_c[dim_c_index];
            dim_c_index += 1;
        }
        else{
            int *section = malloc((2*ts->size[i]+1)*sizeof(int));
            section[0] = ts->size[i];
            for (int j = 0; j < ts->size[i]; ++j){
                section[j*2+1] = j;
                section[j*2+2] = j+1;
            }
            sections[i] = section;
        }
    }
    __slice(m, sections, workspace, dim_c[0]);
}

void __slice(tensor *m, int **sections, float *workspace, int dim)
{
    int *offset = malloc(sizeof(int));
    *offset = 0;
    __slicing(m, sections, workspace, ts->dim, 0, offset, ts->num, dim);
    free(offset);
}

// 每一个区间的第一个值，代表区间分块的数量
void __slicing(tensor *m, int **sections, float *workspace, int dim_now, int offset_o, int *offset_a, int offset, int dim)
{
    int *section = sections[dim_now-1];
    int num = section[0];
    int offset_btensore = (int)(offset / ts->size[dim_now-1]); // 基础偏移量
    for (int i = 0; i < num; ++i){
        int offset_h = section[i*2+1] * offset_btensore + offset_o;
        int size = (section[i*2+2] - section[i*2+1]) * offset_btensore;
        if (dim_now == dim){
            printf("num: %d\n", num);
            memcpy_float_list(workspace, ts->data, *offset_a, offset_h, size);
            *offset_a += size;
        }
        else{
            __slicing(m, sections, workspace, dim_now-1, offset_h, offset_a, offset_btensore, dim);
        }
    }
}

void merge(tensor *m, tensor *n, int dim, int index, float *workspace)
{
    int **sections = malloc(ts->dim*sizeof(int*));
    for (int i = 0; i < dim-1; ++i){
        int *section = malloc(3*sizeof(int));
        section[0] = 1;
        section[1] = 0;
        section[2] = ts->size[i];
        sections[i] = section;
    }
    int *section = malloc(7*sizeof(int));
    section[0] = 3;
    section[1] = 0;
    section[2] = index;
    section[3] = 0;
    section[4] = n->size[dim-1];
    section[5] = index;
    section[6] = ts->size[dim-1];
    sections[dim-1] = section;

    for (int i = dim; i < ts->dim; ++i){
        int *section = malloc((2*ts->size[i]+1)*sizeof(int));
        section[0] = ts->size[i];
        for (int j = 0; j < ts->size[i]; ++j){
            section[2*j+1] = j;
            section[2*j+2] = j+1;
        }
        sections[i] = section;
    }
    int *size = malloc(3*sizeof(int));
    int x = multing_int_list(ts->size, 0, dim-1);
    size[0] = index*x;
    size[1] = n->size[dim-1]*x;
    size[2] = (ts->size[dim-1] - index)*x;
    int *offset = malloc(sizeof(int));
    *offset = 0;
    int *offset_n = malloc(sizeof(int));
    *offset_n = 0;
    __merging(m, n, sections, workspace, ts->dim, 0, offset_n, offset, ts->num, size, dim);
    for (int i = 0; i < ts->dim; ++i){
        free(sections[i]);
    }
    free(sections);
    free(size);
    free(offset);
}

void __merging(tensor *m, tensor *n, int **sections, float *workspace, int dim_now, int offset_m, int *offset_n, int *offset_a, int offset, int *size, int dim)
{
    int *section = sections[dim_now-1];
    int num = section[0];
    int offset_btensore = (int)(offset / ts->size[dim_now-1]);
    for (int i = 0; i < num; ++i){
        int offset_h = section[i*2+1] * offset_btensore + offset_m;
        if (dim_now == dim){
            if (i == 0){
                memcpy_float_list(workspace, ts->data, *offset_a, offset_h, size[i]);
                *offset_a += size[i];
            }
            if (i == 1){
                memcpy_float_list(workspace, n->data, *offset_a, *offset_n, size[i]);
                *offset_n += size[i];
                *offset_a += size[i];
            }
            if (i == 2){
                memcpy_float_list(workspace, ts->data, *offset_a, offset_h, size[i]);
                *offset_a += size[i];
            }
        }
        else{
            __merging(m, n, sections, workspace, dim_now-1, offset_h, offset_n, offset_a, offset_btensore, size, dim);
        }
    }
}

void del(tensor *tensor)
{
    free(tensor->data);
    free(tensor->size);
    free(tensor);
}

float get_sum(tensor *m)
{
    return sum_float_list(ts->data, 0, ts->num);
}

float get_min(tensor *m)
{
    float min = ts->data[0];
    for (int i = 1; i < ts->num; ++i){
        if (ts->data[i] < min) min = ts->data[i];
    }
    return min;
}

float get_max(tensor *m)
{
    float max = ts->data[0];
    for (int i = 1; i < ts->num; ++i){
        if (ts->data[i] > max) max = ts->data[i];
    }
    return max;
}

float get_mean(tensor *m)
{
    return get_sum(m) / ts->num;
}

void saxpy(tensor *mx, tensor *my, float x)
{
    for (int i = 0; i < my->num; ++i){
        mx->data[i] += x * my->data[i];
    }
}

int get_num(tensor *m, float x)
{
    int num = 0;
    for (int i = 0; i < ts->num; ++i){
        if (ts->data[i] == x) num += 1;
    }
    return num;
}

// tensorproxy *init_tensorproxy()
// {
//     tensorproxy *proxy = malloc(sizeof(tensorproxy));
//     proxy->create = create;
//     proxy->copy = copy;
//     proxy->get_lindex = get_lindex;
//     proxy->get_mindex = get_mindex;
//     proxy->get_pixel = get_pixel;
//     proxy->replace_part = replace_part;
//     proxy->resize = resize;
//     proxy->slice = slice;
//     proxy->merge = merge;
//     proxy->del = del;
//     proxy->get_sum = get_sum;
//     proxy->get_min = get_min;
//     proxy->get_max = get_max;
//     proxy->get_mean = get_mean;
//     proxy->get_num = get_num;
//     proxy->saxpy = saxpy;
//     return proxy;
// }