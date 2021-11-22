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
    float *list = malloc((3*3*3)*sizeof(float));
    for (int i = 0; i < 3*3*3; ++i){
        list[i] = i;
    }
    int size[3] = {3, 3, 3};
    Tensor *im = tensor_list(3, size, list);
    tsprint(im);
    int new_size[] = {9, 3};
    resize(im, 2, new_size);
    transposition(im);
    tsprint(im);
    return 0;
}