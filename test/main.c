#include "lumos.h"
#include "network.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    // int h = 4;
    // int w = 4;
    // int c = 3;
    // float *A = calloc(h*w*c, sizeof(float));
    // float *C = calloc(h*w*c, sizeof(float));
    // for (int i = 0; i < h*w; ++i){
    //     A[i] = i+1;
    //     A[h*w+i] = i+1;
    //     A[2*h*w+i] = i+1;
    // }
    // int stride = 1;
    // int pad = 0;
    // int ksize = 3;
    // int height_col = (h + 2*pad - ksize) / stride + 1;
    // int width_col = (w + 2*pad - ksize) / stride + 1;
    // float *B = calloc(height_col*width_col*ksize*ksize*c, sizeof(float));
    // im2col(A, h, w, c, ksize, stride, pad, B);
    // for (int i = 0; i < ksize*ksize*c; ++i){
    //     for (int j = 0; j < height_col*width_col; ++j){
    //         printf("%f ", B[i*height_col*width_col+j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // col2im(B, ksize, stride, pad, h, w, c, C);
    // for (int i = 0; i < c; ++i){
    //     for (int j = 0; j < h*w; ++j){
    //         printf("%f ", C[i*h*w+j]);
    //     }
    //     printf("\n");
    // }

    Network *net = load_network("./cfg/lumos.cfg");
    printf("ok\n");
    return 0;
}