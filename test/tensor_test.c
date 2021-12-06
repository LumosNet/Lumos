#include "lumos.h"
#include "tensor.h"

int main(int argc, char **argv)
{
    int size[] = {1, 2, 3};
    float list[] = {1, 2, 3};
    int index1[] = {0, 0, 0};
    int index2[] = {0, 1, 1};
    int index3[] = {0, 1, 2};
    int **index = malloc(sizeof(int *));
    index[0] = index1;
    index[1] = index2;
    index[2] = index3;
    Tensor *ts = tensor_sparse(3, size, index, list, 3);
    Tensor *tx = tensor_copy(ts);
    tsprint(ts);
    ts_saxpy(ts, tx, 2);
    tsprint(ts);
    free_tensor(ts);
    free_tensor(tx);
    return 0;
}