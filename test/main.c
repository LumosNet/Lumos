#include "lumos.h"
#include "network.h"
#include "utils.h"
#include "data.h"
#include "utils.h"

#include "debug.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv)
{
    Network *net = load_network("./cfg/lumos.cfg");
    init_network(net, "./mnist/mnist.data", NULL);
    train(net);
    train(net);

    // int *h = calloc(1, sizeof(int));
    // int *w = calloc(1, sizeof(int));
    // int *c = calloc(1, sizeof(int));
    // int n = 0;
    // while (1)
    // {
    //     float *data = load_image_data("/home/btboay/MNIST/training/4/42781.png", w, h, c);
    //     printf("%d %f\n", n, data[0]);
    //     printf("------------\n");
    //     n += 1;
    //     // if (n == 10) break;
    // }
    // printf("%d %d %d\n", h[0], w[0], c[0]);
    // for (int k = 0; k < c[0]; ++k){
    //     for (int i = 0; i < h[0]; ++i){
    //         for (int j = 0; j < w[0]; ++j){
    //             printf("%f ", data[k*h[0]*w[0]+i*w[0]+j]);
    //         }
    //         printf("\n");
    //     }
    // }
    return 0;
}