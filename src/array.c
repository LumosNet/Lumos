#include "array.h"

Array *create_array(int row, int col)
{
    int *size = malloc(2*sizeof(int));
    size[0] = col;
    size[1] = row;
    Array *ret = create_matrix(2, size);
    free(size);
    return ret;
}

Array *create_zeros_array(int row, int col)
{
    int *size = malloc(2*sizeof(int));
    size[0] = col;
    size[1] = row;
    Array *ret = create_zeros_matrix(2, size);
    free(size);
    return ret;
}

Array *create_array_full_x(int row, int col, float x)
{
    int *size = malloc(2*sizeof(int));
    size[0] = col;
    size[1] = row;
    Array *ret = create_matrix_full_x(2, size, x);
    free(size);
    return ret;
}

Array *list_to_array(int row, int col, float *list)
{
    int *size = malloc(2*sizeof(int));
    size[0] = col;
    size[1] = row;
    Array *a = list_to_matrix(2, size, list);
    free(size);
    return a;
}

Array *create_unit_array(int row, int col, float x, int flag)
{
    int *size = malloc(2*sizeof(int));
    size[0] = col;
    size[1] = row;
    Array *ret = create_array(row, col);
    int min_row_col = (row <= col) ? row : col;
    for (int i = 0; i < min_row_col; ++i){
        // change_x_in_list(ret->data, y, data_type, i*col+i);
        if (flag) change_float_in_list(ret->data, x, i*col+i);
        else change_float_in_list(ret->data, x, i*col+(col-i-1));
    }
    return ret;
}

Array *copy_array(Array *a)
{
    return copy_matrix(a);
}

// 从0开始的真实数组索引
int replace_aindex_to_lindex(int *size, int row, int col)
{
    int index[] = {col, row};
    return replace_mindex_to_lindex(2, size, index);
}

// 从1开始的物理场景行列索引，且index也必须为物理场景的数组索引
int *replace_lindex_to_aindex(int *size, int index)
{
    int *lindex = replace_lindex_to_mindex(2, size, index);
    int x = lindex[0];
    lindex[0] = lindex[1];
    lindex[1] = x;
    return lindex;
}

float get_pixel_in_array(Array *a, int row, int col)
{
    int index[] = {col, row};
    return get_pixel_in_matrix(a, index);
}

float get_array_min(Array *a)
{
    return get_matrix_min(a);
}

float get_array_max(Array *a)
{
    return get_matrix_max(a);
}

float get_array_mean(Array *a)
{
    return get_matrix_mean(a);
}

int pixel_num_array(Array *a, float x)
{
    return pixel_num_matrix(a, x);
}

void change_pixel_in_array(Array *a, int row, int col, float x)
{
    int index[] = {col, row};
    change_pixel_in_matrix(a, index, x);
}

void resize_array(Array *a, int row, int col)
{
    int *size = malloc(2*sizeof(int));
    size[0] = col;
    size[1] = row;
    resize_matrix(a, 2, size);
    free(size);
}

float *get_row_in_array(Array *a, int row)
{
    int size = a->size[0];
    int offset = (row-1)*size;
    float *data = malloc(size*sizeof(float));
    memcpy_float_list(data, a->data, 0, offset, size);
    return data;
}

float *get_col_in_array(Array *a, int col)
{
    int x = a->size[0];
    int y = a->size[1];
    float *data = malloc(y*sizeof(float));
    for (int i = 0; i < x; ++i){
        data[i] = get_pixel_in_array(a, i+1, col);
    }
    return data;
}

float *get_diagonal_in_array(Array *a, int flag)
{
    int x = a->size[0];
    int y = a->size[1];
    int min = (x<y) ? x : y;
    float *data = malloc(min*sizeof(float));
    for (int i = 0; i < min; ++i){
        if (flag) data[i] = get_pixel_in_array(a, i+1, i+1);
        else data[i] = get_pixel_in_array(a, i+1, min-i);
    }
    return data;
}

Victor *row2Victor(Array *a, int row)
{
    int size = a->size[0];
    int offset = (row-1)*size;
    float *data = malloc(size*sizeof(float));
    memcpy_float_list(data, a->data, 0, offset, size);
    return list_to_Victor(size, 1, data);
}

Victor *col2Victor(Array *a, int col)
{
    int x = a->size[0];
    int y = a->size[1];
    float *data = malloc(y*sizeof(float));
    for (int i = 0; i < x; ++i){
        data[i] = get_pixel_in_array(a, i+1, col);
    }
    return list_to_Victor(y, 0, data);
}

Victor *diagonal2Victor(Array *a, int flag)
{
    int x = a->size[0];
    int y = a->size[1];
    int min = (x<y) ? x : y;
    float *data = malloc(min*sizeof(float));
    for (int i = 0; i < min; ++i){
        if (flag) data[i] = get_pixel_in_array(a, i+1, i+1);
        else data[i] = get_pixel_in_array(a, i+1, min-i);
    }
    return list_to_Victor(min, flag, data);
}

void replace_array_row2list(Array *a, int row, float *list)
{
    int size = a->size[0];
    int index = (row-1)*size;
    memcpy_float_list(a->data, list, index, 0, size);
}

void replace_array_col2list(Array *a, int col, float *list)
{
    for (int i = 0; i < a->size[1]; ++i){
        change_pixel_in_array(a, i+1, col, list[i]);
    }
}

void replace_array_diagonal2list(Array *a, float *list, int flag)
{
    int min = (a->size[0] < a->size[1]) ? a->size[0] : a->size[1];
    for (int i = 0; i < min; ++i){
        if (flag) change_pixel_in_array(a, i+1, i+1, list[i]);
        else change_pixel_in_array(a, i+1, a->size[0]-i, list[i]);
    }
}

void replace_array_row_with_x(Array *a, int row, float x)
{
    for (int i = 0; i < a->size[0]; ++i){
        change_pixel_in_array(a, row, i+1, x);
    }
}

void replace_array_col_with_x(Array *a, int col, float x)
{
    for (int i = 0; i < a->size[1]; ++i){
        change_pixel_in_array(a, i+1, col, x);
    }
}

void replace_diagonal_with_x(Array *a, float x, int flag)
{
    int min = (a->size[0] < a->size[1]) ? a->size[0] : a->size[1];
    for (int i = 0; i < min; ++i){
        if (flag) change_pixel_in_array(a, i+1, i+1, x);
        else change_pixel_in_array(a, i+1, a->size[0]-i, x);
    }
}

void del_row_in_array(Array *a, int row)
{
    a->num -= a->size[0];
    float *data = malloc((a->num)*sizeof(float));
    memcpy_float_list(data, a->data, 0, 0, (row-1)*a->size[0]);
    memcpy_float_list(data, a->data, (row-1)*a->size[0], row*a->size[0], (a->size[1]-row)*a->size[0]);
    free(a->data);
    a->data = data;
    a->size[1] -= 1;
}

void del_col_in_array(Array *a, int col)
{
    a->num -= a->size[1];
    float *data = malloc((a->num)*sizeof(float));
    for (int i = 0; i < a->size[1]; ++i){
        int offset = i*a->size[0];
        memcpy_float_list(data, a->data, i*(a->size[0]-1), offset, col-1);
        memcpy_float_list(data, a->data, i*(a->size[0]-1)+col-1, offset+col, (a->size[0]-col));
    }
    free(a->data);
    a->data = data;
    a->size[0] -= 1;
}

void insert_row_in_array(Array *a, int index, float *data)
{
    a->num += a->size[0];
    float *new_data = malloc(a->num*sizeof(float));
    memcpy_float_list(new_data, a->data, 0, 0, (index-1)*a->size[0]);
    memcpy_float_list(new_data, data, (index-1)*a->size[0], 0, a->size[0]);
    memcpy_float_list(new_data, a->data, index*a->size[0], (index-1)*a->size[0], (a->size[1]-index+1)*a->size[0]);
    free(a->data);
    a->data = new_data;
    a->size[1] += 1;
}

void insert_col_in_array(Array *a, int index, float *data)
{
    a->num += a->size[1];
    float *new_data = malloc((a->num)*sizeof(float));
    for (int i = 0; i < a->size[1]; ++i){
        int offset = i*a->size[0];
        memcpy_float_list(new_data, a->data, i*(a->size[0]+1), offset, index-1);
        new_data[i*(a->size[0]+1)+index-1] = data[i];
        memcpy_float_list(new_data, a->data, i*(a->size[0]+1)+index, offset+index-1, (a->size[0]-index+1));
    }
    free(a->data);
    a->data = new_data;
    a->size[0] += 1;
}

void replace_part_array(Array *a, Array *b, int row, int col)
{
    int *index = malloc(2*sizeof(int));
    index[0] = row;
    index[1] = col;
    replace_part_matrix(a, b, index);
}

// flag=1代表扩充行，flag=0代表扩充列
Array *merge_array(Array *a, Array *b, int flag, int index)
{
    int dim = -1;
    int size[] = {a->size[0], a->size[1]};
    if (flag){
        size[1] += b->size[1];
        dim = 2;
    }
    else{
        size[0] += b->size[0];
        dim = 1;
    }
    Array *ret = create_array(size[1], size[0]);
    merge_matrix(a, b, dim, index, ret->data);
    return ret;
}

Array *slice_array(Array *a, int rowu, int rowd, int coll, int colr)
{
    Array *ret = create_array(rowd-rowu+1, colr-coll+1);
    int offset_row = coll-1;
    int size = colr-coll+1;
    for (int i = rowu-1, n = 0; i < rowd; ++i, ++n){
        int offset_a = i*a->size[0];
        int offset_r = n*ret->size[0];
        memcpy_float_list(ret->data, a->data, offset_r, offset_a+offset_row, size);
    }
    return ret;
}

void del_array(Array *a)
{
    del_matrix(a);
}

void array_overturn_lr(Array *a)
{
    int x = a->size[1];
    int y = a->size[0];
    float *data = malloc(a->num*sizeof(float));
    for (int i = 0; i < x; ++i){
        for (int j = 0; j < y; ++j){
            int col_index = y-1-j;
            data[i*y+j] = a->data[i*y+col_index];
        }
    }
    free(a->data);
    a->data = data;
}

void array_overturn_ud(Array *a)
{
    int x = a->size[1];
    int y = a->size[0];
    float *data = malloc(a->num*sizeof(float));
    for (int i = 0; i < x; ++i){
        int row_index = x-1-i;
        for (int j = 0; j < y; ++j){
            data[i*y+j] = a->data[row_index*y+j];
        }
    }
    free(a->data);
    a->data = data;
}

void array_overturn_diagonal(Array *a, int flag)
{
    float *data = malloc(a->num*sizeof(float));
    for (int i = 0; i < a->size[1]; ++i){
        for (int j = 0; j < a->size[0]; ++j){
            if (flag) data[j*a->size[1]+i] = a->data[i*a->size[0]+j];
            else data[(a->size[0]-j-1)*a->size[1]+a->size[1]-i-1] = a->data[i*a->size[0]+j];
        }
    }
    int x = a->size[0];
    a->size[0] = a->size[1];
    a->size[1] = x;
    free(a->data);
    a->data = data;
}

void array_rotate_left(Array *a, int k)
{
    k %= 4;
    if (k == 0) return;
    if (k == 1){
        int x = a->size[1];
        int y = a->size[0];
        float *data = malloc(x*y*sizeof(float));
        for (int i = 0; i < x; ++i){
            for (int j = 0; j < y; ++j){
                data[(y-j-1)*x+i] = a->data[i*y+j];
            }
        }
        a->size[1] = y;
        a->size[0] = x;
        free(a->data);
        a->data = data;
    }
    if (k == 2){
        array_overturn_ud(a);
        array_overturn_lr(a);
    }
    if (k == 3){
        int x = a->size[1];
        int y = a->size[0];
        float *data = malloc(x*y*sizeof(float));
        for (int i = 0; i < x; ++i){
            for (int j = 0; j < y; ++j){
                data[j*x+x-i-1] = a->data[i*y+j];
            }
        }
        a->size[1] = y;
        a->size[0] = x;
        free(a->data);
        a->data = data;
    }
}

void array_rotate_right(Array *a, int k)
{
    k %= 4;
    if (k == 0) return;
    if (k == 1){
        int x = a->size[1];
        int y = a->size[0];
        float *data = malloc(x*y*sizeof(float));
        for (int i = 0; i < x; ++i){
            for (int j = 0; j < y; ++j){
                data[j*x+x-i-1] = a->data[i*y+j];
            }
        }
        a->size[1] = y;
        a->size[0] = x;
        free(a->data);
        a->data = data;
    }
    if (k == 2){
        array_overturn_ud(a);
        array_overturn_lr(a);
    }
    if (k == 3){
        int x = a->size[1];
        int y = a->size[0];
        float *data = malloc(x*y*sizeof(float));
        for (int i = 0; i < x; ++i){
            for (int j = 0; j < y; ++j){
                data[(y-j-1)*x+i] = a->data[i*y+j];
            }
        }
        a->size[1] = y;
        a->size[0] = x;
        free(a->data);
        a->data = data;
    }
}

void exchange2row_in_array(Array *a, int rowx, int rowy)
{
    int y = a->size[0];
    for (int i = 0; i < y; ++i){
        int n = a->data[(rowx-1)*y+i];
        a->data[(rowx-1)*y+i] = a->data[(rowy-1)*y+i];
        a->data[(rowy-1)*y+i] = n;
    }
}

void exchange2col_in_array(Array *a, int colx, int coly)
{
    int x = a->size[1];
    int y = a->size[0];
    for (int i = 0; i < x; ++i){
        int n = a->data[i*y+colx-1];
        a->data[i*y+colx-1] = a->data[i*y+coly-1];
        a->data[i*y+coly-1] = n;
    }
}

void transposition(Array *a)
{
    array_overturn_diagonal(a, 1);
}

/*
    采用高斯-约旦消元法
    将矩阵右侧拼接一个相同大小的单位阵，构成增广矩阵，通过初等行变化，将原矩阵转化为单位阵，而做相同变化的单位阵，就变成了原矩阵的逆矩阵
*/
Array *array_inverse(Array *a)
{
    int x = a->size[1];
    int y = a->size[0];
    int key = -1;
    Array *res = create_unit_array(x, x, 1, 1);
    for (int i = 0; i < x; ++i) {
        if (a->data[i*y+i] == 0.0) {
            //交换行获得一个非零的对角元素
            for (int j = i + 1; j < x; ++j) {
                if (a->data[j*y+i] != 0.0){
                    key = j;
                    break;
                }
            }
            if (key == -1) {
                return res;
            }
            // 交换矩阵两行
            exchange2row_in_array(a, i+1, key+1);
            exchange2row_in_array(res, i+1, key+1);
        }
        float scalar = 1.0 / a->data[i*y+i];
        array_multiplication_row_x(a, i, scalar);
        array_multiplication_row_x(res, i, scalar);
        for (int k = 0; k < x; ++k) {
            if (i == k) {
                continue;
            }
            float shear_needed = -a->data[k*y+i];
            // 行乘以一个系数加到另一行
            array_rowmulti_add_to_row(a, i, k, shear_needed);
            array_rowmulti_add_to_row(res, i, k, shear_needed);
        }
        key = -1;
    }
    return res;
}

float sum_array(Array *a)
{
    return sum_float_list(a->data, 0, a->num);
}

float get_trace(Array *a)
{
    int x = a->size[1];
    int y = a->size[0];
    int min = (x < y) ? x : y;
    float *diagonal = get_diagonal_in_array(a, 1);
    float res = 0;
    for (int i = 0; i < min; ++i){
        res += diagonal[i];
    }
    return res;
}

Array *array_add(Array *a, Array *b)
{
    int x = a->size[1];
    int y = a->size[0];
    Array *res = create_array(x, y);
    for (int i = 0; i < x; ++i){
        for (int j = 0; j < y; ++j){
            res->data[i*y+j] = a->data[i*y+j] + b->data[i*y+j];
        }
    }
    return res;
}

Array *array_subtract(Array *a, Array *b)
{
    int x = a->size[1];
    int y = a->size[0];
    Array *res = create_array(x, y);
    for (int i = 0; i < x; ++i){
        for (int j = 0; j < y; ++j){
            res->data[i*y+j] = a->data[i*y+j] - b->data[i*y+j];
        }
    }
    return res;
}

Array *array_divide(Array *a, Array *b)
{
    int x = a->size[1];
    int y = a->size[0];
    Array *res = create_array(x, y);
    for (int i = 0; i < x; ++i){
        for (int j = 0; j < y; ++j){
            res->data[i*y+j] = a->data[i*y+j] / b->data[i*y+j];
        }
    }
    return res;
}

Array *array_x_multiplication(Array *a, Array *b)
{
    int x = a->size[1];
    int y = a->size[0];
    Array *res = create_array(x, y);
    for (int i = 0; i < x; ++i){
        for (int j = 0; j < y; ++j){
            res->data[i*y+j] = a->data[i*y+j] * b->data[i*y+j];
        }
    }
    return res;
}

void array_add_x(Array *a, float x)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            a->data[i*n+j] += x;
        }
    }
}

void array_add_row_x(Array *a, int row, float x)
{
    int y = a->size[0];
    for (int i = 0; i < y; ++i){
        a->data[row*y+i] += x;
    }
}

void array_add_col_x(Array *a, int col, float x)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        a->data[i*n+col] += x;
    }
}

void array_subtract_x(Array *a, float x)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            a->data[i*n+j] -= x;
        }
    }
}

void array_subtract_row_x(Array *a, int row, float x)
{
    int y = a->size[0];
    for (int i = 0; i < y; ++i){
        a->data[row*y+i] -= x;
    }
}

void array_subtract_col_x(Array *a, int col, float x)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        a->data[i*n+col] -= x;
    }
}

void array_multiplication_x(Array *a, float x)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            a->data[i*n+j] *= x;
        }
    }
}

void array_multiplication_row_x(Array *a, int row, float x)
{
    int y = a->size[0];
    for (int i = 0; i < y; ++i){
        a->data[row*y+i] *= x;
    }
}

void array_multiplication_col_x(Array *a, int col, float x)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        a->data[i*n+col] *= x;
    }
}

void array_divide_x(Array *a, float x)
{
    if (x == 0) printf("警告，被除数为0\n");
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            a->data[i*n+j] /= x;
        }
    }
}

void array_divide_row_x(Array *a, int row, float x)
{
    if (x == 0) printf("警告，被除数为0\n");
    int y = a->size[0];
    for (int i = 0; i < y; ++i){
        a->data[row*y+i] /= x;
    }
}

void array_divide_col_x(Array *a, int col, float x)
{
    if (x == 0) printf("警告，被除数为0\n");
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        a->data[i*n+col] /= x;
    }
}

void array_add_row_to_row(Array *a, int row1, int row2)
{
    int y = a->size[0];
    for (int i = 0; i < y; ++i){
        a->data[row2*y+i] += a->data[row1*y+i];
    }
}

void array_add_col_to_col(Array *a, int col1, int col2)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        a->data[i*n+col2] += a->data[i*n+col1];
    }
}

void array_rowmulti_add_to_row(Array *a, int row1, int row2, float x)
{
    int y = a->size[0];
    for (int i = 0; i < y; ++i){
        a->data[row2*y+i] += a->data[row1*y+i] * x;
    }
}

void array_colmulti_add_to_col(Array *a, int col1, int col2, float x)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        a->data[i*n+col2] += a->data[i*n+col1] * x;
    }
}

// 采用gemm算法，减少访存次数，从而优化运行时间
Array *gemm(Array *a, Array *b)
{
    int x = a->size[1];
    int y = a->size[0];
    int z = b->size[0];
    Array *res = create_zeros_array(x, z);
    #pragma omp parallel for
    for (int i = 0; i < x; ++i){
        for (int k = 0; k < y; ++k){
            register float temp = a->data[i*y+k];
            for (int j = 0; j < z; ++j){
                res->data[i*z+j] += temp * b->data[k*z+j];
            }
        }
    }
    return res;
}

void array_saxpy(Array *ax, Array *ay, float x)
{
    matrix_saxpy(ax, ay, x);
}

float array_1norm(Array *a)
{
    float res = -9999;
    for (int i = 0; i < a->size[1]; ++i){
        float sum = 0;
        for (int j = 0; j < a->size[0]; ++j){
            sum += get_pixel_in_array(a, j, i);
        }
        if (res < sum){
            res = sum;
        }
    }
    return res;
}

float array_2norm(Array *a)
{
    return 0;
}

float array_infinite_norm(Array *a)
{
    float res = -9999;
    for (int i = 0; i < a->size[0]; ++i){
        float sum = 0;
        for (int j = 0; j < a->size[1]; ++j){
            sum += get_pixel_in_array(a, i, j);
        }
        if (res < sum){
            res = sum;
        }
    }
    return res;
}

float array_frobenius_norm(Array *a)
{
    float res = 0;
    for (int i = 0; i < a->num; ++i){
        res += a->data[i] * a->data[i];
    }
    res = sqrt(res);
    return res;
}

Victor *__householder_v(Victor *x, float *beta)
{
    Victor *m = slice_Victor(x, 1, x->size[0]);
    Victor *mt = copy_victor(m);
    transposition(mt);
    float theta = Victor_x_multiplication(mt, m)->data[0];
    Victor *v = copy_victor(x);
    m->data[0] = 1;
    beta[0] = 0;
    if (theta != 0){
        float u = sqrt(x->data[0]*x->data[0] + theta);
        if (x->data[0] <= 0) v->data[0] = x->data[0] - u;
        else v->data[0] = -theta / (x->data[0] + u);
        beta[0] = (2*v->data[0]*v->data[0]) / (theta + v->data[0]*v->data[0]);
        Victor_divide_x(v, v->data[0]);
    }
    del_Victor(m);
    del_Victor(mt);
    return v;
}

Array *__householder_a(Victor *v, float beta)
{
    Array *In = create_unit_array(v->size[0], v->size[0], 1, 1);
    Array *vt = copy_array(v);
    transposition(vt);
    Array *k = array_x_multiplication(v, vt);
    array_multiplication_x(k, beta);
    Array *P = array_subtract(In, k);
    del_array(In);
    del_array(vt);
    del_array(k);
    return P;
}

Array *householder(Victor *x, float *beta)
{
    Victor *v = __householder_v(x, beta);
    Array *res = __householder_a(v, beta[0]);
    return res;
}

float *givens(float a, float b)
{
    float *res = malloc(2*sizeof(float));
    if (b == 0){
        res[0] = 1;
        res[1] = 0;
    }
    else{
        if (fabs(b) > fabs(a)){
            float x = -a / b;
            res[1] = 1 / (sqrt(1+x*x));
            res[0] = res[1] * x;
        }
        else{
            float x = -b /a;
            res[0] = 1 / (sqrt(1+x*x));
            res[1] = res[0] * x;
        }
    }
}

Array *givens_rotate(Array *a, int i, int k, float c, float s)
{
    Array *res = copy_array(a);
    for (int j = 0; j < res->size[0]; ++j){
        float x = get_pixel_in_array(res, j, i);
        float y = get_pixel_in_array(res, j, k);
        change_pixel_in_array(res, j, i, c*x - s*y);
        change_pixel_in_array(res, j, k, s*x + c*y);
    }
    return res;
}

Array *__householder_QR(Array *a, Array *r, Array *q)
{
    Array *res = copy_array(a);
    for (int i = 0; i < res->size[1]; ++i){
        Array *x = slice_array(res, i+1, res->size[0], i+1, i+1);
        Victor *y = list_to_Victor(x->num, 1, x->data);
        float *beta = malloc(sizeof(float));
        Victor *v = __householder_v(x, beta);
        Array *house = __householder_a(v, beta[0]);

        Array *apart = slice_array(res, i, res->size[0], i, res->size[1]);
        Array *rpart = array_x_multiplication(house, apart);
        replace_part_array(res, rpart, i, i);
        if (i < res->size[0]){
            Victor *vpart = slice_Victor(v, 2, res->size[0]-i+1);
            for (int j = i+1; j < res->size[0]; ++j){
                change_pixel_in_array(res, j, i, vpart->data[j-i-1]);
            }
        }
    }
    return res;
}