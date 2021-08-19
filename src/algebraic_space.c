#include "algebraic_space.h"

AS *create(int dim, int *size, float x)
{
    AS *ret = malloc(sizeof(AS));
    int *m_size = malloc(dim*sizeof(int));
    memcpy(m_size, size, dim*sizeof(int));
    ret->dim = dim;
    ret->size = m_size;
    ret->num = multing_int_list(size, 0, dim);
    ret->data = calloc(ret->num, sizeof(float));
    if (x != 0) full_list_with_float(ret->data, x, ret->num, 0, 0);
    return ret;
}

AS *list_to_AS(int dim, int *size, float *list)
{
    AS *ret = create(dim, size, 0);
    memcpy_float_list(ret->data, list, 0, 0, ret->num);
    return ret;
}

AS *copy(AS *m)
{
    AS *ret = list_to_AS(m->dim, m->size, m->data);
    return ret;
}

int get_lindex(AS *m, int *index)
{
    int ret = 0;
    for (int i = 0; i < m->dim; ++i){
        int x = index[i]-1;
        for (int j = 0; j < i; ++j){
            x *= m->size[j];
        }
        ret += x;
    }
    return ret;
}

int *get_mindex(AS *m, int index)
{
    int *ret = malloc(m->dim*sizeof(int));
    for (int i = m->dim-1; i >= 0; --i){
        int x = 1;
        for (int j = 0; j < i; ++j){
            x *= m->size[j];
        }
        ret[i] = (int)(index / x) + 1;
        index %= x;
    }
    return ret;
}

float get_pixel(AS *m, int *index)
{
    int lindex = get_lindex(m, index);
    return get_float_in_list(m->data, lindex);
}

void change_pixel(AS *m, int *index, float x)
{
    int lindex = get_lindex(m, index);
    change_float_in_list(m->data, x, lindex);
}

void replace_part(AS *m, AS *n, int *index)
{
    int *nindex = malloc(n->dim*sizeof(int));
    full_list_with_int(nindex, 1, n->dim, 1, 0);
    int *mindex = malloc(m->dim*sizeof(int));
    full_list_with_int(mindex, 1, n->dim, 1, 0);
    while (__ergodic(n, nindex)){
        for (int i = 0; i < m->dim; ++i){
            mindex[i] = nindex[i] + index[i] - 1;
        }
        change_pixel(m, mindex, get_pixel(n, nindex));
    }
}

int __ergodic(AS *m, int *index)
{
    int res = 1;
    int dim = m->dim - 1;
    while (index[dim] == m->size[dim]){
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

void resize(AS *m, int dim, int *size)
{
    float *data = m->data;
    m->data = malloc(m->num*sizeof(float));
    m->dim = dim;
    memcpy_float_list(m->data, data, 0, 0, multing_int_list(size, 0, dim));
    memcpy_void_list(m->size, size, INT, 0, 0, m->dim);
    free(data);
}

void slice(AS *m, float *workspace, int *dim_c, int **size_c)
{
    int **sections = malloc(m->dim*sizeof(int*));
    for (int i = 0; i < dim_c[0]; ++i){
        int *section = malloc(3*sizeof(int));
        section[0] = 1;
        section[1] = 0;
        section[2] = m->size[i];
        sections[i] = section;
    }
    int dim_c_index = 0;
    for (int i = dim_c[0]; i < m->dim; ++i){
        if (i == dim_c[dim_c_index]){
            sections[i] = size_c[dim_c_index];
            dim_c_index += 1;
        }
        else{
            int *section = malloc((2*m->size[i]+1)*sizeof(int));
            section[0] = m->size[i];
            for (int j = 0; j < m->size[i]; ++j){
                section[j*2+1] = j;
                section[j*2+2] = j+1;
            }
            sections[i] = section;
        }
    }
    __slice(m, sections, workspace, dim_c[0]);
}

void __slice(AS *m, int **sections, float *workspace, int dim)
{
    int *offset = malloc(sizeof(int));
    *offset = 0;
    __slicing(m, sections, workspace, m->dim, 0, offset, m->num, dim);
    free(offset);
}

// 每一个区间的第一个值，代表区间分块的数量
void __slicing(AS *m, int **sections, float *workspace, int dim_now, int offset_o, int *offset_a, int offset, int dim)
{
    int *section = sections[dim_now-1];
    int num = section[0];
    int offset_base = (int)(offset / m->size[dim_now-1]); // 基础偏移量
    for (int i = 0; i < num; ++i){
        int offset_h = section[i*2+1] * offset_base + offset_o;
        int size = (section[i*2+2] - section[i*2+1]) * offset_base;
        if (dim_now == dim){
            printf("num: %d\n", num);
            memcpy_float_list(workspace, m->data, *offset_a, offset_h, size);
            *offset_a += size;
        }
        else{
            __slicing(m, sections, workspace, dim_now-1, offset_h, offset_a, offset_base, dim);
        }
    }
}

void merge(AS *m, AS *n, int dim, int index, float *workspace)
{
    int **sections = malloc(m->dim*sizeof(int*));
    for (int i = 0; i < dim-1; ++i){
        int *section = malloc(3*sizeof(int));
        section[0] = 1;
        section[1] = 0;
        section[2] = m->size[i];
        sections[i] = section;
    }
    int *section = malloc(7*sizeof(int));
    section[0] = 3;
    section[1] = 0;
    section[2] = index;
    section[3] = 0;
    section[4] = n->size[dim-1];
    section[5] = index;
    section[6] = m->size[dim-1];
    sections[dim-1] = section;

    for (int i = dim; i < m->dim; ++i){
        int *section = malloc((2*m->size[i]+1)*sizeof(int));
        section[0] = m->size[i];
        for (int j = 0; j < m->size[i]; ++j){
            section[2*j+1] = j;
            section[2*j+2] = j+1;
        }
        sections[i] = section;
    }
    int *size = malloc(3*sizeof(int));
    int x = multing_int_list(m->size, 0, dim-1);
    size[0] = index*x;
    size[1] = n->size[dim-1]*x;
    size[2] = (m->size[dim-1] - index)*x;
    int *offset = malloc(sizeof(int));
    *offset = 0;
    int *offset_n = malloc(sizeof(int));
    *offset_n = 0;
    __merging(m, n, sections, workspace, m->dim, 0, offset_n, offset, m->num, size, dim);
    for (int i = 0; i < m->dim; ++i){
        free(sections[i]);
    }
    free(sections);
    free(size);
    free(offset);
}

void __merging(AS *m, AS *n, int **sections, float *workspace, int dim_now, int offset_m, int *offset_n, int *offset_a, int offset, int *size, int dim)
{
    int *section = sections[dim_now-1];
    int num = section[0];
    int offset_base = (int)(offset / m->size[dim_now-1]);
    for (int i = 0; i < num; ++i){
        int offset_h = section[i*2+1] * offset_base + offset_m;
        if (dim_now == dim){
            if (i == 0){
                memcpy_float_list(workspace, m->data, *offset_a, offset_h, size[i]);
                *offset_a += size[i];
            }
            if (i == 1){
                memcpy_float_list(workspace, n->data, *offset_a, *offset_n, size[i]);
                *offset_n += size[i];
                *offset_a += size[i];
            }
            if (i == 2){
                memcpy_float_list(workspace, m->data, *offset_a, offset_h, size[i]);
                *offset_a += size[i];
            }
        }
        else{
            __merging(m, n, sections, workspace, dim_now-1, offset_h, offset_n, offset_a, offset_base, size, dim);
        }
    }
}

void del(AS *AS)
{
    free(AS->data);
    free(AS->size);
    free(AS);
}

float get_sum(AS *m)
{
    return sum_float_list(m->data, 0, m->num);
}

float get_min(AS *m)
{
    float min = m->data[0];
    for (int i = 1; i < m->num; ++i){
        if (m->data[i] < min) min = m->data[i];
    }
    return min;
}

float get_max(AS *m)
{
    float max = m->data[0];
    for (int i = 1; i < m->num; ++i){
        if (m->data[i] > max) max = m->data[i];
    }
    return max;
}

float get_mean(AS *m)
{
    return get_sum(m) / m->num;
}

void saxpy(AS *mx, AS *my, float x)
{
    for (int i = 0; i < my->num; ++i){
        mx->data[i] += x * my->data[i];
    }
}

int get_num(AS *m, float x)
{
    int num = 0;
    for (int i = 0; i < m->num; ++i){
        if (m->data[i] == x) num += 1;
    }
    return num;
}

// ASproxy *init_asproxy()
// {
//     ASproxy *proxy = malloc(sizeof(ASproxy));
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