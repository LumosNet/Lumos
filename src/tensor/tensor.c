#include "tensor.h"

tensor *tensor_x(int dim, int *size, float x)
{
    tensor *ts = malloc(sizeof(tensor));
    int *ts_size = malloc(dim*sizeof(int));
    memcpy(ts_size, size, dim*sizeof(int));
    ts->dim = dim;
    ts->size = ts_size;
    ts->num = multing_int_list(size, 0, dim);
    ts->data = calloc(ts->num, sizeof(float));
    if (x != 0) full_list_with_float(ts->data, x, ts->num, 0, 0);
    return ts;
}

tensor *tensor_list(int dim, int *size, float *list)
{
    tensor *ts = tensor_x(dim, size, 0);
    memcpy_float_list(ts->data, list, 0, 0, ts->num);
    return ts;
}

Tensor *tensor_sparse(int dim, int *size, int **index, float *list, int n)
{
    tensor *ts = tensor_x(dim, size, 0);
    for (int i = 0; i < n; ++i){
        int lindex = index_ts2ls(index[i], dim, size);
        ts->data[lindex] = list[i];
    }
    return ts;
}

void tsprint(tensor *ts)
{
    printf("dimenssion : %d\n", ts->dim);
    printf("instruction : ");
    for (int i = 0; i < ts->dim-1; ++i){
        printf("%d * ", ts->size[i]);
    }
    printf("%d\n", ts->size[ts->dim-1]);
    printf("data num : %d\n", ts->num);
    for (int i = 0; i < ts->num; ++i){
        if ((i+1) % ts->size[0] == 0) printf(" %f\n", ts->data[i]);
        else printf(" %f", ts->data[i]);
        if ((i+1) % (ts->size[0]*ts->size[1]) == 0) printf("\n");
    }
}

int pixel_num(tensor *ts, float x)
{
    int num = 0;
    for (int i = 0; i < ts->num; ++i){
        if (ts->data[i] == x) num += 1;
    }
    return num;
}

tensor *copy(tensor *ts)
{
    tensor *ret_ts = tensor_list(ts->dim, ts->size, ts->data);
    return ret_ts;
}

index_list* get_index(tensor *ts, float x)
{
    index_list *head = NULL;
    index_list *node;
    for (int i = 0; i < ts->num; ++i){
        if (ts->data[i] == x){
            int *ls2ts = index_ls2ts(ts->dim, ts->size, i);
            index_list *index_n = malloc(sizeof(index_list));
            index_n->index = ls2ts;
            index_n->next = NULL;
            if (head == NULL) {
                node = index_n;
                head = node;
            }
            else {
                node->next = index_n;
                node = node->next;
            }
        }
    }
    return head;
}

float get_pixel(tensor *ts, int *index)
{
    // for (int i = 0; i < ts->dim; ++i){
    //     if (index[i] < 1 || index[i] > ts->size[i]){
    //         return 0;
    //     }
    // }
    int ts2ls = index_ts2ls(index, ts->dim, ts->size);
    return ts->data[ts2ls];
}

void change_pixel(tensor *ts, int *index, float x)
{
    // for (int i = 0; i < ts->dim; ++i){
    //     if (index[i] < 1 || index[i] > ts->size[i]) {
    //         return;
    //     }
    // }
    int ts2ls = index_ts2ls(index, ts->dim, ts->size);
    ts->data[ts2ls] = x;
}

void resize(tensor *ts, int dim, int *size)
{
    float *data = ts->data;
    ts->data = malloc(ts->num*sizeof(float));
    ts->dim = dim;
    memcpy_float_list(ts->data, data, 0, 0, multing_int_list(size, 0, dim));
    memcpy_void_list(ts->size, size, INT, 0, 0, ts->dim);
    free(data);
}

void slice(tensor *ts, float *workspace, int *dim_c, int **size_c)
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
    __slice(ts, sections, workspace, dim_c[0]);
}

void merge(tensor *ts, tensor *n, int dim, int index, float *workspace)
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
    __merging(ts, n, sections, workspace, ts->dim, 0, offset_n, offset, ts->num, size, dim);
    for (int i = 0; i < ts->dim; ++i){
        free(sections[i]);
    }
    free(sections);
    free(size);
    free(offset);
}

float get_sum(tensor *ts)
{
    return sum_float_list(ts->data, 0, ts->num);
}

float get_min(tensor *ts)
{
    float min = ts->data[0];
    for (int i = 1; i < ts->num; ++i){
        if (ts->data[i] < min) min = ts->data[i];
    }
    return min;
}

float get_max(tensor *ts)
{
    float max = ts->data[0];
    for (int i = 1; i < ts->num; ++i){
        if (ts->data[i] > max) max = ts->data[i];
    }
    return max;
}

float get_mean(tensor *ts)
{
    return get_sum(ts) / ts->num;
}

void del(tensor *ts)
{
    free(ts->data);
    free(ts->size);
    free(ts);
}

int __ergodic(tensor *ts, int *index)
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

void __slice(tensor *ts, int **sections, float *workspace, int dim)
{
    int *offset = malloc(sizeof(int));
    *offset = 0;
    __slicing(ts, sections, workspace, ts->dim, 0, offset, ts->num, dim);
    free(offset);
}

void __slicing(tensor *ts, int **sections, float *workspace, int dim_now, int offset_o, int *offset_a, int offset, int dim)
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
            __slicing(ts, sections, workspace, dim_now-1, offset_h, offset_a, offset_btensore, dim);
        }
    }
}

void __merging(tensor *ts1, tensor *ts2, int **sections, float *workspace, int dim_now, int offset_m, int *offset_n, int *offset_a, int offset, int *size, int dim)
{
    int *section = sections[dim_now-1];
    int num = section[0];
    int offset_btensore = (int)(offset / ts1->size[dim_now-1]);
    for (int i = 0; i < num; ++i){
        int offset_h = section[i*2+1] * offset_btensore + offset_m;
        if (dim_now == dim){
            if (i == 0){
                memcpy_float_list(workspace, ts1->data, *offset_a, offset_h, size[i]);
                *offset_a += size[i];
            }
            if (i == 1){
                memcpy_float_list(workspace, ts2->data, *offset_a, *offset_n, size[i]);
                *offset_n += size[i];
                *offset_a += size[i];
            }
            if (i == 2){
                memcpy_float_list(workspace, ts1->data, *offset_a, offset_h, size[i]);
                *offset_a += size[i];
            }
        }
        else{
            __merging(ts1, ts2, sections, workspace, dim_now-1, offset_h, offset_n, offset_a, offset_btensore, size, dim);
        }
    }
}