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
#include "umath.h"

#ifdef  __cplusplus
extern "C" {
#endif

Tensor *array_d(int row, int col);
Tensor *array_x(int row, int col, float x);
Tensor *array_list(int row, int col, float *list);
Tensor *array_unit(int row, int col, float x, int flag);
Tensor *array_sparse(int row, int col, int **index, float *list, int n);

// 行列计数从0开始
float ar_get_pixel(Tensor *ts, int row, int col);
void ar_change_pixel(Tensor *ts, int row, int col, float x);
void resize_ar(Tensor *ts, int row, int col);

void replace_rowlist(Tensor *ts, int row, float *list);
void replace_collist(Tensor *ts, int col, float *list);
void replace_diagonalist(Tensor *ts, float *list, int flag);

void replace_rowx(Tensor *ts, int row, float x);
void replace_colx(Tensor *ts, int col, float x);
void replace_diagonalx(Tensor *ts, float x, int flag);

void del_row(Tensor *ts, int row);
void del_col(Tensor *ts, int col);

void insert_row(Tensor *ts, int row, float *data);
void insert_col(Tensor *ts, int col, float *data);

void overturn_lr(Tensor *ts);
void overturn_ud(Tensor *ts);
void overturn_diagonal(Tensor *ts, int flag);

void rotate_left(Tensor *ts, int k);
void rotate_right(Tensor *ts, int k);

void exchange2row(Tensor *ts, int rowx, int rowy);
void exchange2col(Tensor *ts, int colx, int coly);

void row2list(Tensor *ts, int row, float *space);
void col2list(Tensor *ts, int col, float *space);
void diagonal2list(Tensor *ts, int flag, float *space);

void transposition(Tensor *ts);

Tensor *inverse(Tensor *ts);
float trace(Tensor *ts);

void row_addx(Tensor *ts, int row, float x);
void col_addx(Tensor *ts, int col, float x);

void row_multx(Tensor *ts, int row, float x);
void col_multx(Tensor *ts, int col, float x);

void add_row2r(Tensor *ts, int row1, int row2);
void add_col2c(Tensor *ts, int col1, int col2);
void add_multrow2r(Tensor *ts, int row1, int row2, float x);
void add_multcol2c(Tensor *ts, int col1, int col2, float x);

void gemm(Tensor *ts_a, Tensor *ts_b, float *space);

float norm1_ar(Tensor *ts);
float norm2_ar(Tensor *ts);
float infnorm_ar(Tensor *ts);
float fronorm_ar(Tensor *ts);

#ifdef LINEAR
Tensor *householder(Tensor *x, float *beta);
// 给定标量a、b，计算c=cosθ、s=sinθ
float *givens(float a, float b);
Tensor *givens_rotate(Tensor *ts, int i, int k, float c, float s);

Tensor *__householder_v(Tensor *x, float *beta);
Tensor *__householder_a(Tensor *v, float beta);
Tensor *__householder_QR(Tensor *ts, Tensor *r, Tensor *q);
#endif

#ifdef NUMPY
Tensor *merge_array(Tensor *ts, Tensor *b, int dim, int index);
Tensor *slice_array(Tensor *ts, int rowu, int rowd, int coll, int colr);
#endif

#ifdef  __cplusplus
}
#endif

#endif