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
    train(net, 100);
    return 0;
}