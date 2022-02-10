#include <stdio.h>
#include <stdlib.h>

#include "lumos.h"
#include "network.h"

int main(int argc, char **argv)
{
    Network *net = load_network("./cfg/xor.cfg");
    init_network(net, "./demo/xor/xor.data", argv[1]);
    Layer *l = &net->layers[net->n-1];
    test(net, "./demo/xor/data/0_1.png", "./demo/xor/data/0_1.txt");
    printf("xor: [0, 0], test: %f\n", l->input[0]); 
    test(net, "./demo/xor/data/0_2.png", "./demo/xor/data/0_2.txt");
    printf("xor: [1, 1], test: %f\n", l->input[0]);
    test(net, "./demo/xor/data/1_1.png", "./demo/xor/data/1_1.txt");
    printf("xor: [1, 0], test: %f\n", l->input[0]);
    test(net, "./demo/xor/data/1_2.png", "./demo/xor/data/1_2.txt");
    printf("xor: [0, 1], test: %f\n", l->input[0]);
    return 0;
}
