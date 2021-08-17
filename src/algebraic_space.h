#ifndef ALGEBRAIC_SPACE_H
#define ALGEBRAIC_SPACE_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "include.h"
#include "list.h"

int __ergodic(AS *m, int *index);
void __slice(AS *m, int **sections, float *workspace, int dim);
void __slicing(AS *m, int **sections, float *workspace, int dim_now, int offset_o, int *offset_a, int offset, int dim);
void __merging(AS *m, AS *n, int **sections, float *workspace, int dim_now, int offset_m, int *offset_n, int *offset_a, int offset, int *size, int dim);

AS *create(int dim, int *size, float x);

AS *list_to_AS(int dim, int *size, float *list);
AS *copy(AS *m);

int get_lindex(AS *m, int *index);
int *get_mindex(AS *m, int index);

float get_pixel(AS *m, int *index);
void change_pixel(AS *m, int *index, float x);
void replace_part(AS *m, AS *n, int *index);

void resize(AS *m, int dim, int *size);

void slice(AS *m, float *workspace, int *dim_c, int **size_c);
void merge(AS *m, AS *n, int dim, int index, float *workspace);

void del(AS *m);
float get_sum(AS *m);
float get_min(AS *m);
float get_max(AS *m);
float get_mean(AS *m);

void saxpy(AS *mx, AS *my, float x);

int get_num(AS *m, float x);

ASproxy *init_asproxy();
#endif