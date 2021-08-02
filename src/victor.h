#ifndef VICTOR_H
#define VICTOR_H

#include <stdio.h>
#include <stdlib.h>

#include "include.h"
#include "matrix.h"
#include "array.h"

// flag = 1代表为行向量，flag = 0代表列向量

Victor *create_Victor(int num, int flag);
Victor *create_zeros_Victor(int num, int flag);
Victor *create_Victor_full_x(int num, int flag, float x);
Victor *list_to_Victor(int num, int flag, float *list);

Victor *copy_victor(Victor *v);

float get_pixel_in_Victor(Victor *v, int index);
float get_Victor_min(Victor *v);
float get_Victor_max(Victor *v);
float get_Victor_mean(Victor *v);
int pixel_num_Victor(Victor *v, float x);

void change_pixel_in_Victor(Victor *v, int index, float x);

void replace_Victor2list(Victor *v, float *list);
void replace_Victor_with_x(Victor *v, float x);

void del_pixel_in_Victor(Victor *v, int index);
void insert_pixel_in_Victor(Victor *v, int index, float x);

Victor *merge_Victor(Victor *a, Victor *b, int index);
Victor *slice_Victor(Victor *v, int index_h, int index_t);

void del_Victor(Victor *v);

void exchange2pixel_in_Victor(Victor *v, int index_1, int index_2);

float sum_Victor(Victor *v);

Victor *Victor_add(Victor *a, Victor *b);
Victor *Victor_subtract(Victor *a, Victor *b);
Victor *Victor_divide(Victor *a, Victor *b);
Victor *Victor_x_multiplication(Victor *a, Victor *b);

void Victor_add_x(Victor *v, float x);
void Victor_subtract_x(Victor *v, float x);
void Victor_multiplication_x(Victor *v, float x);
void Victor_divide_x(Victor *v, float x);

void Victor_saxpy(Victor *vx, Victor *vy, float x);

#endif