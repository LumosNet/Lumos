#ifndef Vector_H
#define Vector_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lumos.h"
#include "tensor.h"
#include "array.h"

#ifdef  __cplusplus
extern "C" {
#endif

Vector *Vector_x(int num, int flag, float x);
Vector *Vector_list(int num, int flag, float *list);

int colorrow(Vector *v);

float get_pixel_vt(Vector *v, int index);
void change_pixel_vt(Vector *v, int index, float x);
void resize_vt(Vector *v, int num);

void replace_vtlist(Vector *v, float *list);
void replace_vtx(Vector *v, float x);

void del_pixel(Vector *v, int index);
void insert_pixel(Vector *v, int index, float x);

Vector *merge_vt(Vector *a, Vector *b, int index);
Vector *slice_vt(Vector *v, int index_h, int index_t);

float norm1_vt(Vector *v);
float norm2_vt(Vector *v);
float normp_vt(Vector *v, int p);
float infnorm_vt(Vector *v);
float ninfnorm_vt(Vector *v);

#ifdef  __cplusplus
}
#endif

#endif