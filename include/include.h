#ifndef MATRIX_H_API
#define MATRIX_H_API

typedef struct Matrix{
    float *data;
    int dim;
    int *size;
    int num;
} Matrix;

typedef Matrix Array;
typedef Matrix Victor;

#endif