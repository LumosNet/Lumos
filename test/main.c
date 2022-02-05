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
    net->fdebug = fopen("./data/fdebug.data", "wb");
    net->bdebug = fopen("./data/bdebug.data", "wb");
    net->udebug = fopen("./data/fdebug.data", "wb");
    train(net, 8);
    fclose(net->fdebug);
    return 0;
}