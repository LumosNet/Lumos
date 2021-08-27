#include "array.h"

void test_create_array(int row, int col, float x)
{
    Array *a = create_array(row, col, x);
    show_array(a);
}

void test_list_to_array(int row, int col, float *list)
{
    Array *a = list_to_array(row, col, list);
    show_array(a);
}

void test_create_unit_array(int row, int col, float x, int flag)
{
    Array *a = create_unit_array(row, col, x, flag);
    show_array(a);
}

void test_get_array_index(Array *a, int row, int col)
{
    int index = get_array_lindex(a, row, col);
    printf("Index: %d\n", index);
    printf("Pixel: %f\n", a->data[index]);
    int *lindex = get_array_aindex(a, index);
    printf("Index: (%d,%d)\n", lindex[0], lindex[1]);
}

void test_get_array_pixel()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    float pixel = get_array_pixel(a, 3, 2);
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
    Array *a = list_to_array(4, 4, list);
    float min = get_array_mean(a);
    printf("Min Pixel: %f\n", min);
}

void test_pixel_num_array()
{
    float list[] = {\
    1 ,  1,  3,  4, \
    5 ,  6,  6,  8, \
    9 , 11, 11, 12, \
    13, 14, 15, 6  \
    };
    Array *a = list_to_array(4, 4, list);
    int num = pixel_num_array(a, -1);
    printf("num: %d\n", num);
}

void test_change_array_pixel()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    change_array_pixel(a, 4, 4, 0.12);
    show_array(a);
}

void test_resize_array()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    show_array(a);
    resize_array(a, 8, 2);
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
    Array *a = list_to_array(4, 4, list);
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
    Array *a = list_to_array(4, 4, list);
    float *col = get_col_in_array(a, 4);
    printf("Col data: ");
    for (int i = 0; i < a->size[1]; ++i){
        printf("%f ", col[i]);
    }
    printf("\n");
}

void test_row2Victor()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    Victor *v = row2Victor(a, 1);
    show_array(v);
}

void test_col2Victor()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    Victor *v = col2Victor(a, 1);
    show_array(v);
}

void test_diagonal2Victor()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    Victor *v = diagonal2Victor(a, 0);
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
    Array *a = list_to_array(4, 4, list);

    /*
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
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

void test_replace_array_row2list()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    float data[] = {0.1, 0.2, 0.3, 0.4};
    replace_array_row2list(a, 4, data);
    show_array(a);
}

void test_replace_array_col2list()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    float data[] = {0.1, 0.2, 0.3, 0.4};
    replace_array_col2list(a, 1, data);
    show_array(a);
}

void test_replace_array_diagonal2list()
{
    /*
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    */

    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    float data[] = {0.1, 0.2, 0.3, 0.4};
    replace_array_diagonal2list(a, data, 1);
    show_array(a);
}

void test_replace_array_row_with_x()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    replace_array_row_with_x(a, 1, 0.5);
    show_array(a);
}

void test_replace_array_col_with_x()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    replace_array_col_with_x(a, 4, 0.5);
    show_array(a);
}

void test_replace_diagonal_with_x()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);

    /*
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    */
    replace_diagonal_with_x(a, 0.5, 1);
    show_array(a);
}

void test_del_row_in_array()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    del_row_in_array(a, 1);
    show_array(a);
}

void test_del_col_in_array()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    del_col_in_array(a, 4);
    show_array(a);
}

void test_insert_row_in_array()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    float data[] = {0.1,0.2,0.3,0.4};
    insert_row_in_array(a, 1, data);
    show_array(a);
}

void test_insert_col_in_array()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    float data[] = {0.1,0.2,0.3,0.4};
    insert_col_in_array(a, 1, data);
    show_array(a);
}

void test_replace_part_array()
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
    Array *a = list_to_array(4, 4, list1);
    Array *b = list_to_array(2, 3, list2);
    replace_part_array(a, b, 3, 2);
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
    Array *a = list_to_array(4, 4, list_1);

    float list_2[] = {\
    100, 101, 102, 103, \
    104, 105, 106, 107, \
    108, 109, 110, 111, \
    112, 113, 114, 115, \
    116, 117, 118, 119
    };
    Array *b = list_to_array(5, 4, list_2);
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
    Array *a = list_to_array(4, 4, list);
    Array *b = slice_array(a, 2, 2, 1, 4);
    show_array(b);
}

void test_array_overturn_lr()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    array_overturn_lr(a);
    show_array(a);
}

void test_array_overturn_ud()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    array_overturn_ud(a);
    show_array(a);
}

void test_array_overturn_diagonal()
{
    /*
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16  \
    };
    Array *a = list_to_array(4, 4, list);
    */

    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    array_overturn_diagonal(a, 0);
    show_array(a);
}

void test_array_rotate_left()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    array_rotate_left(a, 2);
    show_array(a);
}

void test_array_rotate_right()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    array_rotate_right(a, 2);
    show_array(a);
}

void test_exchange2row_in_array()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    exchange2row_in_array(a, 1, 5);
    show_array(a);
}

void test_exchange2col_in_array()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    exchange2col_in_array(a, 1, 4);
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
    Array *a = list_to_array(5, 4, list);
    transposition(a);
    show_array(a);
}

void test_array_inverse()
{
    
    float list[] = {\
    0,1,2,1,0,3,4,-3,8
    };
    Array *a = list_to_array(3,3, list);
    Array *res = array_inverse(a);
    show_array(res);
}
void test_get_trace()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    float f = get_trace(a);
    show_array(a);
    printf("矩阵的迹 %f",f);
}
void test_array_add()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4 , list);
    show_array(a);
    Array *res =  array_add(a,a);
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
    Array *a = list_to_array(5, 4, list);
    show_array(a);
    printf("相减完成之后的是--\n");
    Array *b = array_subtract(a,a);
    show_array(b);
}
void test_array_divide()
{
     float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(4, 5, list);
    show_array(a);
    Array *b = array_divide(a,a);
    show_array(b);
}
void test_array_x_multiplication()
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
    Array *a = list_to_array(4, 4, lista);
    Array *b = list_to_array(4, 4, listb);
    Array *c = array_x_multiplication(a, b);
    show_array(c);
}

void test_array_add_x()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    array_add_x(a, 1);
    show_array(a);
}

void test_array_add_row_x()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    array_add_row_x(a, 1, 2);
    show_array(a);
}

void test_array_add_col_x()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    array_add_col_x(a, 1, 2);
    show_array(a);
}

void test_array_multiplication_x()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    array_multiplication_x(a, 2);
    show_array(a);
}

void test_array_multiplication_row_x()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    array_multiplication_row_x(a, 1, 2);
    show_array(a);
}

void test_array_multiplication_col_x()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    array_multiplication_col_x(a, 1, 2);
    show_array(a);
}

void test_array_add_row_to_row()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    array_add_row_to_row(a, 1, 2);
    show_array(a);
}

void test_array_add_col_to_col()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    array_add_col_to_col(a, 1, 2);
    show_array(a);
}

void test_array_rowmulti_add_to_row()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    array_rowmulti_add_to_row(a, 1, 2, 2);
    show_array(a);
}

void test_array_colmulti_add_to_col()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    array_colmulti_add_to_col(a, 1, 2 ,2);
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
    Array *a = list_to_array(5, 4, list1);
    Array *b = list_to_array(4, 5, list2);
    Array *res = gemm(a, b);
    show_array(res);
}

void test_array_saxpy()
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
    Array *a = list_to_array(5, 4, list1);
    Array *b = list_to_array(5, 4, list2);
    array_saxpy(a, b, 3);
    show_array(a);
}

void test_array_1norm()
{
    float list[] = {\
    1 ,  2,  3,  4, \
    5 ,  6,  7,  8, \
    9 , 10, 11, 12, \
    13, 14, 15, 16, \
    17, 18, 19, 20  \
    };
    Array *a = list_to_array(5, 4, list);
    float res = array_frobenius_norm(a);
    printf("Array 1-norm: %f\n", res);
}

void test_householder()
{
    float list[] = {\
    1, 2, 3, 4, 5 \
    };
    Victor *v = list_to_Victor(5, 1, list);
    float *beta = malloc(sizeof(float));
    Victor *hv = householder(v, beta);
    printf("Beta: %f\n", beta[0]);
    show_array(hv);
}

int main(int argc, char **argv)
{
    // float list[] = {1,2,3,4,5,6,7,8,9};
    // Array *a = list_to_array(3, 3, list);
    // test_get_array_index(a, 3, 3);
    test_householder();
    return 0;
}