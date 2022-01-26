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
    Network *net = load_network("./cfg/lumos.cfg");
    init_network(net, "./mnist/mnist.data", NULL);
    train(net);
    return 0;
}