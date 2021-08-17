//C实现动态，用到函数指针
#include <stdio.h>
#include <stdlib.h>

#include "include.h"
#include "algebraic_space.h"

int main(int argc, char **argv){
    ASproxy *proxy = init_asproxy();
    int *size = malloc(2*sizeof(int));
    size[0] = 2;
    size[1] = 2;
    AS *matrix_1 = proxy->create(2, size, 1);
    AS *matrix_2 = proxy->copy(matrix_1);
    for (int i = 0; i < size[0]; ++i){
        for (int j = 0; j < size[1]; ++j){
            printf("%f ", matrix->data[i*size[1]+j]);
        }
        printf("\n");
    }
    return 0;
}