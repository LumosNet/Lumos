#include "lumos.h"
#include "tensor.h"
#include "array.h"

int main(int argc, char **argv)
{
    float list[] = {1, 2, 3};
    int index1[] = {0, 0};
    int index2[] = {1, 2};
    int index3[] = {2, 0};
    int **index = malloc(3*sizeof(int *));
    index[0] = index1;
    index[1] = index2;
    index[2] = index3;
    Tensor *ar = array_sparse(3, 3, index, list, 3);
    tsprint(ar);
    add_multcol2c(ar, 0, 1, 2);
    tsprint(ar);
    return 0;
}