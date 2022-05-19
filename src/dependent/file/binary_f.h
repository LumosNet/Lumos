#ifndef BINARY_F_H
#define BINART_F_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
** 三种读取模式
** 0 全读取
** 1 分段读取
** 2 部分读取
*/
#define READALL     0
#define READPART    1
#define READSERIAL  2

/*
**操作文件信息
**0 操作成功
**1-34状态码参考 ferror 返回状态值
**35 模式错误
**36 参数错误
**37 选取范围错误
**38 fseek操作错误
*/
#define SUCESS      0
#define MODERROR    35
#define ARGSERROR   36
#define RANGEERROR  37
#define FSEEKERROR  38
#define FILEEND     39


int write_as_binary(FILE *fp, float* array, size_t size);
int read_as_binary(FILE *fp, int mode, size_t *scop, float**array, size_t *arrsize);

#ifdef __cplusplus
}
#endif

#endif