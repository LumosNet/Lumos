#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "list.h"

void test()
{
    int size[] = {4,4,4};
    float list[] = {\
     0,  1,  2,  3, \
     4,  5,  6,  7, \
     8,  9, 10, 11, \
    12, 13, 14, 15, \
    \
    16, 17, 18, 19, \
    20, 21, 22, 23, \
    24, 25, 26, 27, \
    18, 29, 30, 31, \
    \
    32, 33, 34, 35, \
    36, 37, 38, 39, \
    40, 41, 42, 43, \
    44, 45, 46, 47, \
    \
    48, 49, 50, 51, \
    52, 53, 54, 55, \
    56, 57, 58, 59, \
    60, 61, 62, 63  \
    };

    int size_m[] = {4,2,4};
    float list_m[] = {\
    100, 101, 102, 103, \
    104, 105, 106, 107, \
    \
    108, 109, 110, 111, \
    112, 113, 114, 115, \
    \
    116, 117, 118, 119, \
    120, 121, 122, 123, \
    \
    124, 125, 126, 127, \
    128, 129, 130, 131  \
    };

    tensor *m = tensor_list(3, size, list);
    tensor *n = tensor_list(3, size_m, list_m);

    int num = 4*4*(4+2);
    float *workspace = malloc(num*sizeof(float));
    merge_tensor(m, n, 2, 4, workspace);
    for (int i = 0; i < num; ++i){
        printf("%f\n", workspace[i]);
    }
}

int main(int argc, char **argv)
{
    test();
    return 0;
}