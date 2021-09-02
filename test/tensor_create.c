#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"

int main(int argc, char **argv)
{
    int dim = 4;
    int *size = malloc(dim*sizeof(int));
    for (int i = 0; i < dim; ++i){
        size[i] = i+1;
    }
    tensor * ts = create_x(dim, size, 0.1);
    tsprint(ts);
    int *index = malloc(ts->dim*sizeof(int));
    for (int i = 0; i < ts->dim; ++i){
        index[i] = 0;
    }
    test_gan(ts, index);
    return 0;
}