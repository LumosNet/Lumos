#ifndef CPU_H
#define CPU_H

#ifdef __cplusplus
extern "C"{
#endif

// offset=1 为正常偏移
void fill_cpu(float *data, int len, float x, int offset);
void multy_cpu(float *data, int len, float x, int offset);
void add_cpu(float *data, int len, float x, int offset);

#ifdef __cplusplus
}
#endif

#endif