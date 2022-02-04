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
#include <math.h>

int main(int argc, char **argv)
{
    // Network *net = load_network("./cfg/xor.cfg");
    // init_network(net, "./XOR/xor.data", NULL);
    // printf("255\n");
    // train(net, 5);
    float a[] = {1, 2, 3, 4, 5};
    float b[] = {.1, .2, .3, .4, .5};
    saxpy(a, b, 5, -0.1, a);
    for (int i = 0; i < 5; ++i){
        printf("%f ", a[i]);
    }
    printf("\n");
    return 0;
}