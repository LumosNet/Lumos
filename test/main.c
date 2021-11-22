#include "image.h"
#include "gray_process.h"
#include "array.h"
#include "im2col.h"
#include "parser.h"
#include "utils.h"
#include "network.h"
#include "lumos.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    float *list = malloc((4*4*3)*sizeof(float));
    for (int i = 0; i < 4*4*3; ++i){
        list[i] = i;
    }
    int size[3] = {4, 4, 3};
    Tensor *im = tensor_list(3, size, list);
    tsprint(im);
    float weights[] = {1, 1, 1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 0, 0, 0, -1, -1, -1, \
                    2, 2, 2, 0, 0, 0, -2, -2, -2, 2, 2, 2, 0, 0, 0, -2, -2, -2, 2, 2, 2, 0, 0, 0, -2, -2, -2, \
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    Array *weight = array_list(3, 27, weights);
    transposition(weight);
    tsprint(weight);
    Image *new = convolutional(im, weight, 0, 1);
    tsprint(new);
    return 0;
}