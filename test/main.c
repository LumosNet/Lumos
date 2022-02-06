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
    init_network(net, "./XOR/xor.data", "./data/w.weights");
    net->fdebug = fopen("./data/fdebug.data", "wb");
    net->bdebug = fopen("./data/bdebug.data", "wb");
    net->udebug = fopen("./data/udebug.data", "wb");
    // train(net, 20000);
    debug_str(net->fdebug, "\ntesting\n");
    test(net, "./XOR/data/0_1.png", "./XOR/data/0_1.txt");
    test(net, "./XOR/data/0_2.png", "./XOR/data/0_2.txt");
    test(net, "./XOR/data/1_1.png", "./XOR/data/1_1.txt");
    test(net, "./XOR/data/1_2.png", "./XOR/data/1_2.txt");
    fclose(net->fdebug);
    return 0;
}