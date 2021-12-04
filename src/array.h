#ifndef ARRAY_H
#define ARRAY_H

/*************************************************************************************************\
 * 描述
	实现对Tensor的基本处理（操作）
     Tensor是在tensor基础上的特殊分支
     矩阵在整个框架中具有举足轻重的地位
     未来会单独编写CUDA代码来加速矩阵的运算

     在这里，我们定义了矩阵的创建、元素获取、矩阵计算等内容
     可以将其定义为一个轻量级的线性矩阵计算库
\*************************************************************************************************/
#include <math.h>

#include "lumos.h"
#include "tensor.h"
#include "vector.h"

#ifdef  __cplusplus
extern "C" {
#endif

Tensor *array(int row, int col);
Tensor *array_x(int row, int col, float x);
Tensor *array_list(int row, int col, float *list);
Tensor *array_unit(int row, int col, float x, int flag);
Tensor *array_sparse(int row, int col, int **index, float *list);

// 行列计数从0开始
float ar_get_pixel(Tensor *a, int row, int col);
void ar_change_pixel(Tensor *a, int row, int col, float x);
void resize_ar(Tensor *a, int row, int col);

Tensor *row2Tensor(Tensor *a, int row);
Tensor *col2Tensor(Tensor *a, int col);
Tensor *diagonal2Tensor(Tensor *a, int flag);

void replace_rowlist(Tensor *a, int row, float *list);
void replace_collist(Tensor *a, int col, float *list);
void replace_diagonalist(Tensor *a, float *list, int flag);

void replace_rowx(Tensor *a, int row, float x);
void replace_colx(Tensor *a, int col, float x);
void replace_diagonalx(Tensor *a, float x, int flag);

void del_row(Tensor *a, int row);
void del_col(Tensor *a, int col);

void insert_row(Tensor *a, int index, float *data);
void insert_col(Tensor *a, int index, float *data);

void replace_part(Tensor *a, Tensor *b, int row, int col);

Tensor *merge_array(Tensor *a, Tensor *b, int dim, int index);
Tensor *slice_array(Tensor *a, int rowu, int rowd, int coll, int colr);

void overturn_lr(Tensor *a);
void overturn_ud(Tensor *a);
void overturn_diagonal(Tensor *a, int flag);

void rotate_left(Tensor *a, int k);
void rotate_right(Tensor *a, int k);

void exchange2row(Tensor *a, int rowx, int rowy);
void exchange2col(Tensor *a, int colx, int coly);

void transposition(Tensor *a);

Tensor *inverse(Tensor *a);
float trace(Tensor *a);

void row_addx(Tensor *a, int row, float x);
void col_addx(Tensor *a, int col, float x);

void row_multx(Tensor *a, int row, float x);
void col_multx(Tensor *a, int col, float x);
void add_row2r(Tensor *a, int row1, int row2);
void add_col2c(Tensor *a, int col1, int col2);
void add_multrow2r(Tensor *a, int row1, int row2, float x);
void add_multcol2c(Tensor *a, int col1, int col2, float x);

Tensor *gemm(Tensor *a, Tensor *b);

float norm1_ar(Tensor *a);
float norm2_ar(Tensor *a);
float infnorm_ar(Tensor *a);
float fronorm_ar(Tensor *a);

Tensor *householder(Tensor *x, float *beta);
// 给定标量a、b，计算c=cosθ、s=sinθ
float *givens(float a, float b);
Tensor *givens_rotate(Tensor *a, int i, int k, float c, float s);

Tensor *__householder_v(Tensor *x, float *beta);
Tensor *__householder_a(Tensor *v, float beta);
Tensor *__householder_QR(Tensor *a, Tensor *r, Tensor *q);

#ifdef  __cplusplus
}
#endif

#endif