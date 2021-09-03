#ifndef ARRAY_H
#define ARRAY_H

/*************************************************************************************************\
 * 描述
	实现对Array的基本处理（操作）
     Array是在tensor基础上的特殊分支
     矩阵在整个框架中具有举足轻重的地位
     未来会单独编写CUDA代码来加速矩阵的运算

     在这里，我们定义了矩阵的创建、元素获取、矩阵计算等内容
     可以将其定义为一个轻量级的线性矩阵计算库
\*************************************************************************************************/
#include "tensor.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct base_arop{
     Victor*  (*row2Victor)();
     Victor*  (*col2Victor)();
     Victor*  (*diagonal2Victor)();

     void     (*replace_rowlist)();
     void     (*replace_collist)();
     void     (*replace_diagonallist)();

     void     (*replace_rowx)();
     void     (*replace_colx)();
     void     (*replace_diagonalx)();

     void     (*del_row)();
     void     (*del_col)();

     void     (*insert_row)();
     void     (*insert_col)();

     void     (*replace_part)();

     Array*   (*merge_array)();
     Array*   (*slice_array)();

     void     (*overturn_lr)();
     void     (*overturn_ud)();
     void     (*overturn_diagonal)();

     void     (*rotate_left)();
     void     (*rotate_right)();

     void     (*exchange2row)();
     void     (*exchange2col)();

     void     (*transposition)();
} base_arop, BaseAROp;

typedef struct numeric_op{
     Array*   (*add_ar)();
     Array*   (*subtract_ar)();
     Array*   (*divide_ar)();
     Array*   (*multiply_ar)();

     void     (*array_add_x)();
     void     (*array_add_row_x)();
     void     (*array_add_col_x)();

     void     (*array_multiply_x)();
     void     (*array_multiply_row_x)();
     void     (*array_multiply_col_x)();

     void     (*array_add_row_to_row)();
     void     (*array_add_col_to_col)();
     void     (*array_rowmulti_add_to_row)();
     void     (*array_colmulti_add_to_col)();

     float    (*trace)();

     Array*   (*gemm)();
     Array*   (*array_inverse)();
     void     (*array_saxpy)();

     float    (*array_1norm)();
     float    (*array_2norm)();
     float    (*array_infinite_norm)();
     float    (*array_frobenius_norm)();
} numeric_op, NumericOp;

Array *array_x(int row, int col, float x);
Array *array_list(int row, int col, float *list);
Array *array_unit(int row, int col, float x, int flag);
Array *array_sparse(int row, int col, int **index, float *list);

float get_pixel_ar(Array *a, int row, int col);
void change_pixel_ar(Array *a, int row, int col, float x);
void resize_ar(Array *a, int row, int col);

Victor *row2Victor(Array *a, int row);
Victor *col2Victor(Array *a, int col);
Victor *diagonal2Victor(Array *a, int flag);

void replace_rowlist(Array *a, int row, float *list);
void replace_collist(Array *a, int col, float *list);
void replace_diagonallist(Array *a, float *list, int flag);

void replace_rowx(Array *a, int row, float x);
void replace_colx(Array *a, int col, float x);
void replace_diagonalx(Array *a, float x, int flag);

void del_row(Array *a, int row);
void del_col(Array *a, int col);

void insert_row(Array *a, int index, float *data);
void insert_col(Array *a, int index, float *data);

void replace_part(Array *a, Array *b, int row, int col);

Array *merge_array(Array *a, Array *b, int dim, int index);
Array *slice_array(Array *a, int rowu, int rowd, int coll, int colr);

void overturn_lr(Array *a);
void overturn_ud(Array *a);
void overturn_diagonal(Array *a, int flag);

void rotate_left(Array *a, int k);
void rotate_right(Array *a, int k);

void exchange2row(Array *a, int rowx, int rowy);
void exchange2col(Array *a, int colx, int coly);

void transposition(Array *a);

Array *array_inverse(Array *a);
float sum_array(Array *a);
float get_trace(Array *a);
Array *array_add(Array *a, Array *b);
Array *array_subtract(Array *a, Array *b);
Array *array_divide(Array *a, Array *b);
Array *array_x_multiply(Array *a, Array *b);
void array_add_x(Array *a, float x);
void array_add_row_x(Array *a, int row, float x);
void array_add_col_x(Array *a, int col, float x);
void array_multiply_x(Array *a, float x);
void array_multiply_row_x(Array *a, int row, float x);
void array_multiply_col_x(Array *a, int col, float x);
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

Victor *__householder_v(Victor *x, float *beta);
Array *__householder_a(Victor *v, float beta);
Array *__householder_QR(Array *a, Array *r, Array *q);

#ifdef  __cplusplus
}
#endif

#endif