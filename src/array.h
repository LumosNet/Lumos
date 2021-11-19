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
#include <math.h>

#include "lumos.h"
#include "tensor.h"
#include "Vector.h"

#ifdef  __cplusplus
extern "C" {
#endif

Array *array_x(int row, int col, float x);
Array *array_list(int row, int col, float *list);
Array *array_unit(int row, int col, float x, int flag);
Array *array_sparse(int row, int col, int **index, float *list);

float get_pixel_ar(Array *a, int row, int col);
void change_pixel_ar(Array *a, int row, int col, float x);
void resize_ar(Array *a, int row, int col);

Vector *row2Vector(Array *a, int row);
Vector *col2Vector(Array *a, int col);
Vector *diagonal2Vector(Array *a, int flag);

void replace_rowlist(Array *a, int row, float *list);
void replace_collist(Array *a, int col, float *list);
void replace_diagonalist(Array *a, float *list, int flag);

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

Array *inverse(Array *a);
float trace(Array *a);
Array *add_ar(Array *a, Array *b);
Array *subtract_ar(Array *a, Array *b);
Array *divide_ar(Array *a, Array *b);
Array *multiply_ar(Array *a, Array *b);
void add_arx(Array *a, float x);
void row_addx(Array *a, int row, float x);
void col_addx(Array *a, int col, float x);
void array_multx(Array *a, float x);
void row_multx(Array *a, int row, float x);
void col_multx(Array *a, int col, float x);
void add_row2r(Array *a, int row1, int row2);
void add_col2c(Array *a, int col1, int col2);
void add_multrow2r(Array *a, int row1, int row2, float x);
void add_multcol2c(Array *a, int col1, int col2, float x);
Array *gemm(Array *a, Array *b);
void saxpy(Array *ax, Array *ay, float x);
float norm1_ar(Array *a);
float norm2_ar(Array *a);
float infnorm_ar(Array *a);
float fronorm_ar(Array *a);

Array *householder(Vector *x, float *beta);
// 给定标量a、b，计算c=cosθ、s=sinθ
float *givens(float a, float b);
Array *givens_rotate(Array *a, int i, int k, float c, float s);

Vector *__householder_v(Vector *x, float *beta);
Array *__householder_a(Vector *v, float beta);
Array *__householder_QR(Array *a, Array *r, Array *q);

#ifdef  __cplusplus
}
#endif

#endif