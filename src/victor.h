#ifndef VICTOR_H
#define VICTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tensor.h"

#ifdef  __cplusplus
extern "C" {
#endif

// flag = 1代表为行向量，flag = 0代表列向量
Victor *victor_x(int num, int flag, float x);
Victor *victor_list(int num, int flag, float *list);

float get_pixel_vt(Victor *v, int index);
void change_pixel_vt(Victor *v, int index, float x);

void replace_victorlist(Victor *v, float *list);
void replace_victorx(Victor *v, float x);

void del_pixel(Victor *v, int index);
void insert_pixel(Victor *v, int index, float x);

Victor *merge_victor(Victor *a, Victor *b, int index);
Victor *slice_victor(Victor *v, int index_h, int index_t);

Victor *add_vt(Victor *a, Victor *b);
Victor *subtract_vt(Victor *a, Victor *b);
Victor *divide_vt(Victor *a, Victor *b);
Victor *multiply_vt(Victor *a, Victor *b);

void Victor_addx(Victor *v, float x);
void Victor_multx(Victor *v, float x);

float norm1_vt(Victor *v);
float norm2_vt(Victor *v);
float normp_vt(Victor *v, int p);
float infnorm_vt(Victor *v);
float ninfnorm_vt(Victor *v);

#ifdef  __cplusplus
}
#endif

#endif