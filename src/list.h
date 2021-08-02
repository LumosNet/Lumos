#ifndef LIST_H
#define LIST_H

/*************************************************************************************************\
 * 描述
       对不同类型的对象进行操作
       包含所有C语言支持的基本数值类型，分别为
       char, unsigned char, short, unsigned short, int, unsigned int, long
       unsigned long, float, double, long int, long long, long double
       该文件内容为matrix库的最基础依赖
 * 内容
       主要实现内容为
       1.不同数据类型的内存空间申请
       2.不同数据类型内存空间中内容的赋值
       3.不同数据类型内存空间中内容的修改
       4.对不同数据类型的内存空间的索引
       5.对不同数据类型的内存空间进行拷贝
       6.对不同数据类型的内存空间中的数据进行累加
       7.对不同数据类型的内存空间中的数据进行累乘
\*************************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef enum DataType{
    CHAR, UNSIGNED_CHAR, SHORT, UNSIGNED_SHORT, INT, UNSIGNED_INT,\
    LONG, UNSIGNED_LONG, FLOAT, DOUBLE, LONG_INT, LONG_LONG, LONG_DOUBLE
} DataType;

/*************************************************************************************************\
 * 描述
       为不同类型创建内存空间
 * 参数
       num:数量
       data_type:数据类型，DataType为一个枚举类型，包含所有C语言支持的数值类型
                     详细信息请查看matrix.h中的相关定义
 * 返回
       void*
       返回创建好的内存空间指针
\*************************************************************************************************/
void *create_memory(int num, DataType data_type);

/*************************************************************************************************\
 * 描述
       为不同类型数组填充数据
 * 参数
       origin:待填充的数组（地址）
       x:想要填充的数据
       data_type:数据类型，DataType为一个枚举类型，包含所有C语言支持的数值类型
                     详细信息请查看matrix.h中的相关定义
       num:待填充数组的大小
       stride:填充步长（每隔stride进行一次填充）
 * 返回
\*************************************************************************************************/
void full_list_with_x(void *origin, void *x, DataType data_type, int num, int stride, int flag);

/*************************************************************************************************\
 * 描述
       十三个函数，对应十三种不同的数据类型，实现对应类型数组数据的填充
       十三种数据类型分别为:  CHAR, UNSIGNED_CHAR, SHORT, UNSIGNED_SHORT, INT, UNSIGNED_INT,
                            LONG, UNSIGNED_LONG, FLOAT, DOUBLE, LONG_INT, LONG_LONG, LONG_DOUBLE
 * 参数
       origin:待填充的数组（地址）
       x:想要填充的数据
       data_type:数据类型，DataType为一个枚举类型，包含所有C语言支持的数值类型
                     详细信息请查看matrix.h中的相关定义
       num:待填充数组的大小
       stride:填充步长（每隔stride进行一次填充）
       flag:当flag=0 stride为赋值后步长，当flag=1 stride为赋值前步长
 * 返回
\*************************************************************************************************/
void full_list_with_char(char *origin, char x, int num, int stride, int flag);                                         // 填充char类型数组
void full_list_with_unsigned_char(unsigned char *origin, unsigned char x, int num, int stride, int flag);              // 填充unsigned char数组
void full_list_with_short(short *origin, short x, int num, int stride, int flag);                                      // 填充short类型数组
void full_list_with_unsigned_short(unsigned short *origin, unsigned short x, int num, int stride, int flag);           // 填充填充unsigned short类型数组
void full_list_with_int(int *origin, int x, int num, int stride, int flag);                                            // 填充int类型数组
void full_list_with_unsigned_int(unsigned int *origin, unsigned int x, int num, int stride, int flag);                 // 填充unsigned int类型数组
void full_list_with_long(long *origin, long x, int num, int stride, int flag);                                         // 填充long类型数组
void full_list_with_unsigned_long(unsigned long *origin, unsigned long x, int num, int stride, int flag);              // 填充unsigned long类型数组
void full_list_with_float(float *origin, float x, int num, int stride, int flag);                                      // 填充float类型数组
void full_list_with_double(double *origin, double x, int num, int stride, int flag);                                   // 填充double类型数组
void full_list_with_long_int(long int *origin, long int x, int num, int stride, int flag);                             // 填充long int类型数组
void full_list_with_long_long(long long *origin, long long x, int num, int stride, int flag);                          // 填充long long类型数组
void full_list_with_long_double(long double *origin, long double x, int num, int stride, int flag);                    // 填充long double类型数组

/*************************************************************************************************\
 * 描述
       十三个函数，对应十三种不同的数据类型，实现对应类型数组数据的修改
       十三种数据类型分别为:  CHAR, UNSIGNED_CHAR, SHORT, UNSIGNED_SHORT, INT, UNSIGNED_INT,
                            LONG, UNSIGNED_LONG, FLOAT, DOUBLE, LONG_INT, LONG_LONG, LONG_DOUBLE
 * 参数
       origin:待修改的数组（地址）
       x:想要修改的数据
       data_type:数据类型，DataType为一个枚举类型，包含所有C语言支持的数值类型
                     详细信息请查看matrix.h中的相关定义
       index:修改数据的位置索引
 * 返回
\*************************************************************************************************/
void change_pixel_in_list(void *origin, void *x, DataType data_type, int index);

void change_char_in_list(char *origin, char x, int index);                                                             // 填充char类型数组
void change_unsigned_char_in_list(unsigned char *origin, unsigned char x, int index);                                  // 填充unsigned char数组
void change_short_in_list(short *origin, short x, int index);                                                          // 填充short类型数组
void change_unsigned_short_in_list(unsigned short *origin, unsigned short x, int index);                               // 填充填充unsigned short类型数组
void change_int_in_list(int *origin, int x, int index);                                                                // 填充int类型数组
void change_unsigned_int_in_list(unsigned int *origin, unsigned int x, int index);                                     // 填充unsigned int类型数组
void change_long_in_list(long *origin, long x, int index);                                                             // 填充long类型数组
void change_unsigned_long_in_list(unsigned long *origin, unsigned long x, int index);                                  // 填充unsigned long类型数组
void change_float_in_list(float *origin, float x, int index);                                                          // 填充float类型数组
void change_double_in_list(double *origin, double x, int index);                                                       // 填充double类型数组
void change_long_int_in_list(long int *origin, long int x, int index);                                                 // 填充long int类型数组
void change_long_long_in_list(long long *origin, long long x, int index);                                              // 填充long long类型数组
void change_long_double_in_list(long double *origin, long double x, int index);                                        // 填充long double类型数组

/*************************************************************************************************\
 * 描述
       十三个函数，对应十三种不同的数据类型，实现对应类型数组数据的获取
       十三种数据类型分别为:  CHAR, UNSIGNED_CHAR, SHORT, UNSIGNED_SHORT, INT, UNSIGNED_INT,
                            LONG, UNSIGNED_LONG, FLOAT, DOUBLE, LONG_INT, LONG_LONG, LONG_DOUBLE
 * 参数
       origin:待获取的数组（地址）
       data_type:数据类型，DataType为一个枚举类型，包含所有C语言支持的数值类型
                     详细信息请查看matrix.h中的相关定义
       index:获取数据的位置索引
 * 返回
       void*
       获取到的数据内存地址，获取到的数据存放在新申请的内存空间中
       请在使用完该数据后，将其内存进行释放
\*************************************************************************************************/
void *get_pixel_in_list(void *origin, DataType data_type, int index);

char get_char_in_list(char *origin, int index);                                                                        // 获取char类型数组
unsigned char get_unsigned_char_in_list(unsigned char *origin, int index);                                             // 获取unsigned char数组
short get_short_in_list(short *origin, int index);                                                                     // 获取short类型数组
unsigned short get_unsigned_short_in_list(unsigned short *origin, int index);                                          // 获取填充unsigned short类型数组
int get_int_in_list(int *origin, int index);                                                                           // 获取int类型数组
unsigned int get_unsigned_int_in_list(unsigned int *origin, int index);                                                // 获取unsigned int类型数组
long get_long_in_list(long *origin, int index);                                                                        // 获取long类型数组
unsigned long get_unsigned_long_in_list(unsigned long *origin, int index);                                             // 获取unsigned long类型数组
float get_float_in_list(float *origin, int index);                                                                     // 获取float类型数组
double get_double_in_list(double *origin, int index);                                                                  // 获取double类型数组
long int get_long_int_in_list(long int *origin, int index);                                                            // 获取long int类型数组
long long get_long_long_in_list(long long *origin, int index);                                                         // 获取long long类型数组
long double get_long_double_in_list(long double *origin, int index);                                                   // 获取long double类型数组

/*************************************************************************************************\
 * 描述
       十三个函数，对应十三种不同的数据类型，实现对应类型数组数据的拷贝
       十三种数据类型分别为:  CHAR, UNSIGNED_CHAR, SHORT, UNSIGNED_SHORT, INT, UNSIGNED_INT,
                            LONG, UNSIGNED_LONG, FLOAT, DOUBLE, LONG_INT, LONG_LONG, LONG_DOUBLE
 * 参数
       ret:拷贝数据存放内存地址
       origin:待拷贝的数组（地址）
       data_type:数据类型，DataType为一个枚举类型，包含所有C语言支持的数值类型
                     详细信息请查看matrix.h中的相关定义
       offset_r:拷贝数据存放地址偏移量
       offset_o:拷贝数据源数组的偏移量
       num:拷贝数据数量
 * 返回
 * 补充
                     index_r              num
              _ _ _ _ _/\_ _ _ _  _ _ _ _/\_ _ _ _ _
             /                  \/                  \
       ret:[][][][][][][][][][][][][][][][][][][][][][][][][][][]

                     index_o              num
                _ _ _ _ _/\_ _ _ _  _ _ _ _/\_ _ _ _ _
               /                  \/                  \
       origin:[][][][][][][][][][][][][][][][][][][][][][][][][][][]
\*************************************************************************************************/
void memcpy_void_list(void *ret, void *origin, DataType data_type, int offset_r, int offset_o, int num);

void memcpy_char_list(char *ret, char *origin, int offset_r, int offset_o, int num);                                   // 拷贝char类型数组
void memcpy_unsigned_char_list(unsigned char *ret, unsigned char *origin, int offset_r, int offset_o, int num);        // 拷贝unsigned char数组
void memcpy_short_list(short *ret, short *origin, int offset_r, int offset_o, int num);                                // 拷贝short类型数组
void memcpy_unsigned_short_list(unsigned short *ret, unsigned short *origin, int offset_r, int offset_o, int num);     // 拷贝填充unsigned short类型数组
void memcpy_int_list(int *ret, int *origin, int offset_r, int offset_o, int num);                                      // 拷贝int类型数组
void memcpy_unsigned_int_list(unsigned int *ret, unsigned int *origin, int offset_r, int offset_o, int num);           // 拷贝unsigned int类型数组
void memcpy_long_list(long *ret, long *origin, int offset_r, int offset_o, int num);                                   // 拷贝long类型数组
void memcpy_unsigned_long_list(unsigned long *ret, unsigned long *origin, int offset_r, int offset_o, int num);        // 拷贝unsigned long类型数组
void memcpy_float_list(float *ret, float *origin, int offset_r, int offset_o, int num);                                // 拷贝float类型数组
void memcpy_double_list(double *ret, double *origin, int offset_r, int offset_o, int num);                             // 拷贝double类型数组
void memcpy_long_int_list(long int *ret, long int *origin, int offset_r, int offset_o, int num);                       // 拷贝long int类型数组
void memcpy_long_long_list(long long *ret, long long *origin, int offset_r, int offset_o, int num);                    // 拷贝long long类型数组
void memcpy_long_double_list(long double *ret, long double *origin, int offset_r, int offset_o, int num);              // 拷贝long double类型数组

/*************************************************************************************************\
 * 描述
       十三个函数，对应十三种不同的数据类型，对数组数据进行累加
       十三种数据类型分别为:  CHAR, UNSIGNED_CHAR, SHORT, UNSIGNED_SHORT, INT, UNSIGNED_INT,
                            LONG, UNSIGNED_LONG, FLOAT, DOUBLE, LONG_INT, LONG_LONG, LONG_DOUBLE
 * 参数
       origin:待累加的数组（地址）
       data_type:数据类型，DataType为一个枚举类型，包含所有C语言支持的数值类型
                     详细信息请查看matrix.h中的相关定义
       index:数组长度
 * 返回
       void*
       返回累加结果
 * 补充
       char和unsigned char类型数组累加结果分别为int和unsigned int类型
\*************************************************************************************************/
void *sum_list(void *origin, DataType data_type, int offset, int len);

int sum_char_list(char *origin, int offset, int len);                                                                       // 累加char类型数组
unsigned int sum_unsigned_char_list(unsigned char *origin, int offset, int len);                                            // 累加unsigned char数组
short sum_short_list(short *origin, int offset, int len);                                                                   // 累加short类型数组
unsigned short sum_unsigned_short_list(unsigned short *origin, int offset, int len);                                        // 累加填充unsigned short类型数组
int sum_int_list(int *origin, int offset, int len);                                                                         // 累加int类型数组
unsigned int sum_unsigned_int_list(unsigned int *origin, int offset, int len);                                              // 累加unsigned int类型数组
long sum_long_list(long *origin, int offset, int len);                                                                      // 累加long类型数组
unsigned long sum_unsigned_long_list(unsigned long *origin, int offset, int len);                                           // 累加unsigned long类型数组
float sum_float_list(float *origin, int offset, int len);                                                                   // 累加float类型数组
double sum_double_list(double *origin, int offset, int len);                                                                // 累加double类型数组
long int sum_long_int_list(long int *origin, int offset, int len);                                                          // 累加long int类型数组
long long sum_long_long_list(long long *origin, int offset, int len);                                                       // 累加long long类型数组
long double sum_long_double_list(long double *origin, int offset, int len);                                                 // 累加long double类型数组

/*************************************************************************************************\
 * 描述
       十三个函数，对应十三种不同的数据类型，对数组数据进行累乘
       十三种数据类型分别为:  CHAR, UNSIGNED_CHAR, SHORT, UNSIGNED_SHORT, INT, UNSIGNED_INT,
                            LONG, UNSIGNED_LONG, FLOAT, DOUBLE, LONG_INT, LONG_LONG, LONG_DOUBLE
 * 参数
       origin:待累乘的数组（地址）
       data_type:数据类型，DataType为一个枚举类型，包含所有C语言支持的数值类型
                     详细信息请查看matrix.h中的相关定义
       index:数组长度
 * 返回
       void*
       返回累乘结果
 * 补充
       char和unsigned char类型数组累乘结果分别为int和unsigned int类型
\*************************************************************************************************/
void *multing_list(void *origin, DataType data_type, int offset, int len);

int multing_char_list(char *origin, int offset, int len);                                                                   // 累乘char类型数组
unsigned int multing_unsigned_char_list(unsigned char *origin, int offset, int len);                                        // 累乘unsigned char数组
short multing_short_list(short *origin, int offset, int len);                                                               // 累乘short类型数组
unsigned short multing_unsigned_short_list(unsigned short *origin, int offset, int len);                                    // 累乘填充unsigned short类型数组
int multing_int_list(int *origin, int offset, int len);                                                                     // 累乘int类型数组
unsigned int multing_unsigned_int_list(unsigned int *origin, int offset, int len);                                          // 累乘unsigned int类型数组
long multing_long_list(long *origin, int offset, int len);                                                                  // 累乘long类型数组
unsigned long multing_unsigned_long_list(unsigned long *origin, int offset, int len);                                       // 累乘unsigned long类型数组
float multing_float_list(float *origin, int offset, int len);                                                               // 累乘float类型数组
double multing_double_list(double *origin, int offset, int len);                                                            // 累乘double类型数组
long int multing_long_int_list(long int *origin, int offset, int len);                                                      // 累乘long int类型数组
long long multing_long_long_list(long long *origin, int offset, int len);                                                   // 累乘long long类型数组
long double multing_long_double_list(long double *origin, int offset, int len);                                             // 累乘long double类型数组

#endif