#include "array.h"

void test_array_x(int row, int col, float x)
{
    Tensor *a = array_x(row, col, x);
    show_array(a);
}

void test_array_list(int row, int col, float *list)
{
    Tensor *a = array_list(row, col, list);
    show_array(a);
}

void test_array_unit(int row, int col, float x, int flag)
{
    Tensor *a = array_unit(row, col, x, flag);
    show_array(a);
}

void test_get_array_index(Tensor *ts, int row, int col)
{
    int index = get_array_lindex(a, row, col);
    printf("Index: %d\n", index);
    printf("Pixel: %f\n", a->data[index]);
    int *lindex = get_array_aindex(a, index);
    printf("Index: (%d,%d)\n", lindex[0], lindex[1]);
}

void test_ts_get_pixel_ar()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    float pixel = ts_get_pixel_ar(a, 3, 2);
    printf("Pixel: %f\n", pixel);
}

void test_get_array_min()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    float min = get_array_mean(a);
    printf("Min Pixel: %f\n", min);
}

void test_ts_pixel_num_array()
{
    float list[] = {\
    1 ,  1,  3,  4, \
    5 ,  6,  6,  8, \
    9 , 11, 11, 12, \
    13, 14, 15, 6  \
    };
    Tensor *a = array_list(4, 4, list);
    int num = ts_pixel_num_array(a, -1);
    printf("num: %d\n", num);
}

void test_ts_change_pixel_ar()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    ts_change_pixel_ar(a, 4, 4, 0.12);
    show_array(a);
}

void test_resize_ar()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    show_array(a);
    resize_ar(a, 8, 2);
    show_array(a);
}

void test_get_row_in_array()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    float *row = get_row_in_array(a, 4);
    printf("Row data: ");
    for (int i = 0; i < a->size[0]; ++i){
        printf("%f ", row[i]);
    }
    printf("\n");
}

void test_get_col_in_array()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    float *col = get_col_in_array(a, 4);
    printf("Col data: ");
    for (int i = 0; i < a->size[1]; ++i){
        printf("%f ", col[i]);
    }
    printf("\n");
}

void test_row2Tensor()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    Tensor *v = row2Tensor(a, 1);
    show_array(v);
}

void test_col2Tensor()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    Tensor *v = col2Tensor(a, 1);
    show_array(v);
}

void test_diagonal2Tensor()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    Tensor *v = diagonal2Tensor(a, 0);
    show_array(v);
}

void test_get_diagonal_in_array()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);

    /*
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    */
    show_array(a);
    float *diagonal = get_diagonal_in_array(a, 0);
    int min = (a->size[0] < a->size[1]) ? a->size[0] : a->size[1];
    printf("Diagonal data: ");
    for (int i = 0; i < min; ++i){
        printf("%f ", diagonal[i]);
    }
    printf("\n");
}

void test_replace_rowlist()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    float data[] = {0.1, 0.2, 0.3, 0.4};
    replace_rowlist(a, 4, data);
    show_array(a);
}

void test_replace_collist()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    float data[] = {0.1, 0.2, 0.3, 0.4};
    replace_collist(a, 1, data);
    show_array(a);
}

void test_replace_diagonalist()
{
    /*
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    */

    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    float data[] = {0.1, 0.2, 0.3, 0.4};
    replace_diagonalist(a, data, 1);
    show_array(a);
}

void test_replace_rowx()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    replace_rowx(a, 1, 0.5);
    show_array(a);
}

void test_replace_colx()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    replace_colx(a, 4, 0.5);
    show_array(a);
}

void test_replace_diagonalx()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);

    /*
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    */
    replace_diagonalx(a, 0.5, 1);
    show_array(a);
}

void test_del_row()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    del_row(a, 1);
    show_array(a);
}

void test_del_col()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    del_col(a, 4);
    show_array(a);
}

void test_insert_row()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    float data[] = {0.1,0.2,0.3,0.4};
    insert_row(a, 1, data);
    show_array(a);
}

void test_insert_col()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    float data[] = {0.1,0.2,0.3,0.4};
    insert_col(a, 1, data);
    show_array(a);
}

void test_replace_part()
{
    float list1[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };

    float list2[] = {
    0.1 , 0.2, 0.3, \
    0.4 , 0.5, 0.6  \
    };
    Tensor *a = array_list(4, 4, list1);
    Tensor *b = array_list(2, 3, list2);
    replace_part(a, b, 3, 2);
    show_array(a);
}

void test_merge_array()
{
    float list_1[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list_1);

    float list_2[] = {\
    100, 101, 102, 103, \
    104, 105, 106, 107, \
    108, 109, 110, 111, \
    112, 113, 114, 115, \
    116, 117, 118, 119
    };
    Tensor *b = array_list(5, 4, list_2);
    tensor *c = merge_array(a, b, 1, 0);
    show_array(c);
}

void test_slice_array()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    Tensor *b = slice_array(a, 2, 2, 1, 4);
    show_array(b);
}

void test_overturn_lr()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    overturn_lr(a);
    show_array(a);
}

void test_overturn_ud()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    overturn_ud(a);
    show_array(a);
}

void test_overturn_diagonal()
{
    /*
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, list);
    */

    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    overturn_diagonal(a, 0);
    show_array(a);
}

void test_rotate_left()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    rotate_left(a, 2);
    show_array(a);
}

void test_rotate_right()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    rotate_right(a, 2);
    show_array(a);
}

void test_exchange2row()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    exchange2row(a, 1, 5);
    show_array(a);
}

void test_exchange2col()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    exchange2col(a, 1, 4);
    show_array(a);
}

void test_transposition()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    transposition(a);
    show_array(a);
}

void test_inverse()
{
    
    float list[] = {\
    0,1,2,1,0,3,4,-3,8
    };
    Tensor *a = array_list(3,3, list);
    Tensor *res = inverse(a);
    show_array(res);
}
void test_trace()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    float f = trace(a);
    show_array(a);
    printf("矩阵的迹 %f",f);
}
void test_add_ar()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4 , list);
    show_array(a);
    Tensor *res =  add_ar(a,a);
    show_array(res);
}
void test_array_substract()
{
     float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    show_array(a);
    printf("相减完成之后的是--\n");
    Tensor *b = subtract_ar(a,a);
    show_array(b);
}
void test_divide_ar()
{
     float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(4, 5, list);
    show_array(a);
    Tensor *b = divide_ar(a,a);
    show_array(b);
}
void test_multiply_ar()
{
    float lista[] = {\
     1,  2,  3,  4, \
     5,  6,  7,  8, \
     9, 10, 11, 12, \
    13, 14, 15, 16  \
    };
    float listb[] = {\
     1,  2,  3,  4, \
     5,  6,  7,  8, \
     9, 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Tensor *a = array_list(4, 4, lista);
    Tensor *b = array_list(4, 4, listb);
    Tensor *c = multiply_ar(a, b);
    show_array(c);
}

void test_add_arx()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    add_arx(a, 1);
    show_array(a);
}

void test_row_addx()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    row_addx(a, 1, 2);
    show_array(a);
}

void test_col_addx()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    col_addx(a, 1, 2);
    show_array(a);
}

void test_array_multx()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    array_multx(a, 2);
    show_array(a);
}

void test_row_multx()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    row_multx(a, 1, 2);
    show_array(a);
}

void test_col_multx()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    col_multx(a, 1, 2);
    show_array(a);
}

void test_add_row2r()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    add_row2r(a, 1, 2);
    show_array(a);
}

void test_add_col2c()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    add_col2c(a, 1, 2);
    show_array(a);
}

void test_add_multrow2r()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    add_multrow2r(a, 1, 2, 2);
    show_array(a);
}

void test_add_multcol2c()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    add_multcol2c(a, 1, 2 ,2);
    show_array(a);
}

void test_gemm()
{
    float list1[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    float list2[] = {\
    1,  2,  3,  4,  5, \
    6,  7,  8,  9, 10, \
    11, 12, 13, 14,15, \
    16, 17, 18, 19,20  \
    };
    Tensor *a = array_list(5, 4, list1);
    Tensor *b = array_list(4, 5, list2);
    Tensor *res = gemm(a, b);
    show_array(res);
}

void test_saxpy()
{
    float list1[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    float list2[] = {\
    1,  2,  3,  4,  5, \
    6,  7,  8,  9, 10, \
    11, 12, 13, 14,15, \
    16, 17, 18, 19,20  \
    };
    Tensor *a = array_list(5, 4, list1);
    Tensor *b = array_list(5, 4, list2);
    saxpy(a, b, 3);
    show_array(a);
}

void test_norm1_ar()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Tensor *a = array_list(5, 4, list);
    float res = fronorm_ar(a);
    printf("Tensor 1-norm: %f\n", res);
}

void test_householder()
{
    float list[] = {\
    1, 2, 3, 4, 5 \
    };
    Tensor *v = Tensor_list(5, 1, list);
    float *beta = malloc(sizeof(float));
    Tensor *hv = householder(v, beta);
    printf("Beta: %f\n", beta[0]);
    show_array(hv);
}

int main(int argc, char **argv)
{
    // float list[] = {1,2,3,4,5,6,7,8,9};
    // Tensor *a = array_list(3, 3, list);
    // test_get_array_index(a, 3, 3);
    test_householder();
    return 0;
}