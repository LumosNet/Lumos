#ifndef ARRAY_H
#define ARRAY_H

/*************************************************************************************************\
 * 描述
	实现对Array的基本处理（操作）
     Array是在AS基础上的特殊分支
     矩阵在整个框架中具有举足轻重的地位
     未来会单独编写CUDA代码来加速矩阵的运算

     在这里，我们定义了矩阵的创建、元素获取、矩阵计算等内容
     可以将其定义为一个轻量级的线性矩阵计算库
\*************************************************************************************************/
#include "include.h"
#include "algebraic_space.h"
#include "victor.h"

Victor *__householder_v(Victor *x, float *beta);
Array *__householder_a(Victor *v, float beta);
Array *__householder_QR(Array *a, Array *r, Array *q);
Array *create_array(int row, int col, float x);
Array *list_to_array(int row, int col, float *list);
Array *create_unit_array(int row, int col, float x, int flag);
Array *copy_array(Array *a);
int get_array_lindex(Array *a, int row, int col);
int *get_array_aindex(Array *a, int index);
float get_array_pixel(Array *a, int row, int col);
float get_array_min(Array *a);
float get_array_max(Array *a);
float get_array_mean(Array *a);
int pixel_num_array(Array *a, float x);
void change_array_pixel(Array *a, int row, int col, float x);
void resize_array(Array *a, int row, int col);
float *get_row_in_array(Array *a, int row);
float *get_col_in_array(Array *a, int col);
float *get_diagonal_in_array(Array *a, int flag);
Victor *row2Victor(Array *a, int row);
Victor *col2Victor(Array *a, int col);
Victor *diagonal2Victor(Array *a, int flag);
void replace_array_row2list(Array *a, int row, float *list);
void replace_array_col2list(Array *a, int col, float *list);
void replace_array_diagonal2list(Array *a, float *list, int flag);
void replace_array_row_with_x(Array *a, int row, float x);
void replace_array_col_with_x(Array *a, int col, float x);
void replace_diagonal_with_x(Array *a, float x, int flag);
void del_row_in_array(Array *a, int row);
void del_col_in_array(Array *a, int col);
void insert_row_in_array(Array *a, int index, float *data);
void insert_col_in_array(Array *a, int index, float *data);
void replace_part_array(Array *a, Array *b, int row, int col);
Array *merge_array(Array *a, Array *b, int dim, int index);
Array *slice_array(Array *a, int rowu, int rowd, int coll, int colr);
void del_array(Array *a);
void array_overturn_lr(Array *a);
void array_overturn_ud(Array *a);
void array_overturn_diagonal(Array *a, int flag);
void array_rotate_left(Array *a, int k);
void array_rotate_right(Array *a, int k);
void exchange2row_in_array(Array *a, int rowx, int rowy);
void exchange2col_in_array(Array *a, int colx, int coly);
void transposition(Array *a);
Array *array_inverse(Array *a);
float sum_array(Array *a);
float get_trace(Array *a);
Array *array_add(Array *a, Array *b);
Array *array_subtract(Array *a, Array *b);
Array *array_divide(Array *a, Array *b);
Array *array_x_multiplication(Array *a, Array *b);
void array_add_x(Array *a, float x);
void array_add_row_x(Array *a, int row, float x);
void array_add_col_x(Array *a, int col, float x);
void array_multiplication_x(Array *a, float x);
void array_multiplication_row_x(Array *a, int row, float x);
void array_multiplication_col_x(Array *a, int col, float x);
void array_add_row_to_row(Array *a, int row1, int row2);
void array_add_col_to_col(Array *a, int col1, int col2);
void array_rowmulti_add_to_row(Array *a, int row1, int row2, float x);
void array_colmulti_add_to_col(Array *a, int col1, int col2, float x);
Array *gemm(Array *a, Array *b);
void array_saxpy(Array *ax, Array *ay, float x);
float array_1norm(Array *a);
float array_2norm(Array *a);
float array_infinite_norm(Array *a);
float array_frobenius_norm(Array *a);
Array *householder(Victor *x, float *beta);
// 给定标量a、b，计算c=cosθ、s=sinθ
float *givens(float a, float b);
Array *givens_rotate(Array *a, int i, int k, float c, float s);

void show_array(Array *a);
#endif