#ifndef CPU_CALL_H
#define CPU_CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cpu.h"

#ifdef  __cplusplus
extern "C" {
#endif

void call_fill_cpu(void **params, void **ret);
void call_multy_cpu(void **params, void **ret);
void call_add_cpu(void **params, void **ret);

void call_min_cpu(void **params, void **ret);
void call_max_cpu(void **params, void **ret);
void call_sum_cpu(void **params, void **ret);
void call_mean_cpu(void **params, void **ret);

void call_matrix_add_cpu(void **params, void **ret);
void call_matrix_subtract_cpu(void **params, void **ret);
void call_matrix_multiply_cpu(void **params, void **ret);
void call_matrix_divide_cpu(void **params, void **ret);

void call_saxpy_cpu(void **params, void **ret);
void call_sum_channel_cpu(void **params, void **ret);

void call_one_hot_encoding(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
