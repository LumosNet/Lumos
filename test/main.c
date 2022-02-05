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
    Network *net = load_network("./cfg/xor.cfg");
    init_network(net, "./XOR/xor.data", NULL);
    printf("255\n");
    train(net, 200);
    // float a[] = {0, 0};
    // float b[] = {255, 255};
    // float c[] = {0, 255};
    // float d[] = {255, 0};
    // save_image_data(a, 2, 1, 1, "./XOR/data/0_1.png");
    // save_image_data(b, 2, 1, 1, "./XOR/data/0_2.png");
    // save_image_data(c, 2, 1, 1, "./XOR/data/1_1.png");
    // save_image_data(d, 2, 1, 1, "./XOR/data/1_2.png");
    return 0;
}