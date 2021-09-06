#ifndef VICTOR_H
#define VICTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tensor.h"
#include "array.h"

#ifdef  __cplusplus
extern "C" {
#endif

Victor *victor_x(int num, int flag, float x);
Victor *victor_list(int num, int flag, float *list);

int colorrow(Victor *v);

float get_pixel_vt(Victor *v, int index);
void change_pixel_vt(Victor *v, int index, float x);
void resize_vt(Victor *v, int num);

void replace_vtlist(Victor *v, float *list);
void replace_vtx(Victor *v, float x);

void del_pixel(Victor *v, int index);
void insert_pixel(Victor *v, int index, float x);

Victor *merge_vt(Victor *a, Victor *b, int index);
Victor *slice_vt(Victor *v, int index_h, int index_t);

float norm1_vt(Victor *v);
float norm2_vt(Victor *v);
float normp_vt(Victor *v, int p);
float infnorm_vt(Victor *v);
float ninfnorm_vt(Victor *v);

#ifdef  __cplusplus
}
#endif

#endif