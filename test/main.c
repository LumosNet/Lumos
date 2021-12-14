#include "array.h"
#include "tensor.h"
#include "vector.h"
#include "umath.h"
#include "lumos.h"
#include "gemm.h"
#include "image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    float *A = calloc(16, sizeof(float));
    for (int i = 0; i < 16; ++i){
        A[i] = i+1;
    }
    float *B = malloc(9*sizeof(float));
    im2col(A, 4, 4, 1, 3, 1, 0, B);
    for (int i = 0; i < 9; ++i){
        printf("%f ", B[i]);
    }
    printf("\n");
    return 0;
}