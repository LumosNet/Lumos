#ifndef UTEST_H
#define UTEST_H

#include <stdio.h>
#include <stdlib.h>

#define SUCCESS            0
#define ERROR_NUM          1
#define ERROR_MEM_APPLY    2
#define ERROR_MEM_SET      3
#define ERROR_PARAM_INIT   4

#ifdef __cplusplus
extern "C" {
#endif

void test_run(char *msg);
void test_res(int flag, char *msg);
void test_msg(char *msg);

#ifdef __cplusplus
}
#endif

#endif