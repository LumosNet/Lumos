#ifndef Tensor_H
#define Tensor_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lumos.h"
#include "tensor.h"
#include "array.h"

#ifdef  __cplusplus
extern "C" {
#endif

Tensor *Tensor_x(int num, int flag, float x);
Tensor *Tensor_list(int num, int flag, float *list);

int colorrow(Tensor *v);

float ts_get_pixel_vt(Tensor *v, int index);
void ts_change_pixel_vt(Tensor *v, int index, float x);
void resize_vt(Tensor *v, int num);

void replace_vtlist(Tensor *v, float *list);
void replace_vtx(Tensor *v, float x);

void del_pixel(Tensor *v, int index);
void insert_pixel(Tensor *v, int index, float x);

Tensor *merge_vt(Tensor *a, Tensor *b, int index);
Tensor *slice_vt(Tensor *v, int index_h, int index_t);

float norm1_vt(Tensor *v);
float norm2_vt(Tensor *v);
float normp_vt(Tensor *v, int p);
float infnorm_vt(Tensor *v);
float ninfnorm_vt(Tensor *v);

#ifdef  __cplusplus
}
#endif

#endif