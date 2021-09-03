#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"

int main(int argc, char **argv)
{
    int dim = 3;
    int size[] = {3, 2, 4};
    int **index = malloc(3*sizeof(int*));
    int index1[] = {1, 2, 2, 1};
    int index2[] = {3, 1, 1, 1};
    int index3[] = {2, 2, 3, 2};
    index[0] = index1;
    index[1] = index2;
    index[2] = index3;
    float list[] = {20, 30, 40};
    tensor * ts = tensor_sparse(dim, size, index, list, 3);

    int dim2 = 4;
    int size2[] = {2, 2, 3, 2};
    tsprint(ts);
    resize(ts, dim2, size2);
    tsprint(ts);
    return 0;
}