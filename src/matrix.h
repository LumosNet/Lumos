#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "include.h"
#include "list.h"

Matrix *create_matrix(int dim, int *size);
Matrix *create_zeros_matrix(int dim, int *size);
Matrix *create_matrix_full_x(int dim, int *size, float x);
Matrix *list_to_matrix(int dim, int *size, float *list);
Matrix *copy_matrix(Matrix *m);

int replace_mindex_to_lindex(int dim, int *size, int *index);
int *replace_lindex_to_mindex(int dim, int *size, int index);

float get_pixel_in_matrix(Matrix *m, int *index);
void change_pixel_in_matrix(Matrix *m, int *index, float x);
void replace_part_matrix(Matrix *m, Matrix *n, int *index);
int __ergodic_matrix(Matrix *m, int *index);

void resize_matrix(Matrix *m, int dim, int *size);

void slice_matrix(Matrix *m, float *workspace, int *dim_c, int **size_c);
void merge_matrix(Matrix *m, Matrix *n, int dim, int index, float *workspace);

void del_matrix(Matrix *m);
float sum_matrix(Matrix *m);
float get_matrix_min(Matrix *m);
float get_matrix_max(Matrix *m);
float get_matrix_mean(Matrix *m);

void matrix_saxpy(Matrix *mx, Matrix *my, float x);

int pixel_num_matrix(Matrix *m, float x);

void __slice(Matrix *m, int **sections, float *workspace, int dim);
void __slicing(Matrix *m, int **sections, float *workspace, int dim_now, int offset_o, int *offset_a, int offset, int dim);

void __merging(Matrix *m, Matrix *n, int **sections, float *workspace, int dim_now, int offset_m, int *offset_n, int *offset_a, int offset, int *size, int dim);

#endif