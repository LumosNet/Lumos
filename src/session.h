#ifndef SESSION_H
#define SESSION_H

#include "tensor.h"
#include "array.h"
#include "Vector.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct session{
    tensor*   (*copy)();
    void      (*tsprint)();
    int       (*pixel_num)();
    int       (*get_index)();
    float     (*get_pixel)();
    void      (*change_pixel)();
    void      (*resize)();
    void      (*slice)();
    void      (*merge)();
    float     (*get_sum)();
    float     (*get_min)();
    float     (*get_max)();
    float     (*get_mean)();
    void      (*del)();
} session, Session;

typedef struct base_arop{
    Vector*  (*row2Vector)();
    Vector*  (*col2Vector)();
    Vector*  (*diagonal2Vector)();

    void     (*replace_rowlist)();
    void     (*replace_collist)();
    void     (*replace_diagonalist)();

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

typedef struct numeric_arop{
    Array*   (*add_ar)();
    Array*   (*subtract_ar)();
    Array*   (*divide_ar)();
    Array*   (*multiply_ar)();

    void     (*add_arx)();
    void     (*row_addx)();
    void     (*col_addx)();

    void     (*array_multx)();
    void     (*row_multx)();
    void     (*col_multx)();

    void     (*add_row2r)();
    void     (*add_col2c)();
    void     (*add_multrow2r)();
    void     (*add_multcol2c)();

    float    (*trace)();

    Array*   (*gemm)();
    Array*   (*inverse)();
    void     (*saxpy)();

    float    (*norm1_ar)();
    float    (*norm2_ar)();
    float    (*infnorm_ar)();
    float    (*fronorm_ar)();
} numeric_arop, NumericAROp;

typedef struct transform_arop
{
    float*   (*givens)();
    Array*   (*givens_rotate)();
    Array*   (*householder)();
} transform_arop, TransformAROp;

typedef struct base_vtop
{
    void      (*replace_vtlist)();
    void      (*replace_vtx)();

    void      (*del_pixel)();
    void      (*insert_pixel)();

    Vector*   (*merge_vt)();
    Vector*   (*slice_vt)();
} base_vtop, BaseVTOP;

typedef struct numeric_vtop
{
    float     (*norm1_vt)();
    float     (*norm2_vt)();
    float     (*normp_vt)();
    float     (*infnorm_vt)();
    float     (*ninfnorm_vt)();
} numeric_vtop, NumericVTOP;


#ifdef  __cplusplus
}
#endif

#endif