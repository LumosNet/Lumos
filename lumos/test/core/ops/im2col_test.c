#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utest.h"
#include "im2col.h"

void test_im2col()
{
    test_run("test_im2col");
    float *img = malloc(32*sizeof(float));
    float res[72] = {
        1,   2,   5,   6, \
        2,   3,   6,   7, \
        3,   4,   7,   8, \
        5,   6,   9,   10, \
        6,   7,   10,  11, \
        7,   8,   11,  12, \
        9,   10,  13,  14, \
        10,  11,  14,  15, \
        11,  12,  15,  16, \
        0.1,  0.2,  0.5,  0.6, \
        0.2,  0.3,  0.6,  0.7, \
        0.3,  0.4,  0.7,  0.8, \
        0.5,  0.6,  0.9,  1.0, \
        0.6,  0.7,  1.0,  1.1, \
        0.7,  0.8,  1.1,  1.2, \
        0.9,  1.0,  1.3,  1.4, \
        1.0,  1.1,  1.4,  1.5, \
        1.1,  1.2,  1.5,  1.6
    };
    float *space = malloc(72*sizeof(float));
    for (int i = 0; i < 16; ++i){
        img[i] = i+1;
        img[i+16] = 0.1*(i+1);
    }
    im2col(img, 4, 4, 2, 3, 1, 0, space);
    for (int i = 0; i < 72; ++i){
        if (fabs(res[i]-space[i]) > 1e-6){
            test_res(1, "");
            return;
        }
    }
    test_res(0, "");
}

int main()
{
    test_im2col();
    return 0;
}
