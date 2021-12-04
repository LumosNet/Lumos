#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"
#include "array.h"
#include "vector.h"

#include "loss.h"

int main(int argc, char **argv)
{
    // float list1[] = {1, 2, 3};
    // float list2[] = {-1, -2, -3};
    // Tensor *v1 = Tensor_list(3, 1, list1);
    // Tensor *v2 = Tensor_list(3, 1, list2);
    // float res = mse(v1, v2);
    // printf("%f\n", res);

    int size[] = {4, 3, 2, 2};
    tensor *ts = tensor_x(4, size, 1);
    int index[] = {3, 2, 1, 2};
    index_ts2ls(index, ts->dim, ts->size);
    ts_change_pixel(ts, index, 14);
    tsprint(ts);
    return 0;
}