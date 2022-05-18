#include <stdio.h>
#include <stdlib.h>

#include "lumos.h"
#include "network.h"

void test_xor_demo(char **argv)
{
    Network *net = load_network("./cfg/xor.cfg");
    init_network(net, "./demo/xor/xor.data", argv[2]);
    Layer *l = &net->layers[net->n-1];
    test(net, "./demo/xor/data/00.png", "./demo/xor/data/00.txt");
    printf("xor: [0, 0], test: %f\n", l->input[0]);
    test(net, "./demo/xor/data/11.png", "./demo/xor/data/11.txt");
    printf("xor: [1, 1], test: %f\n", l->input[0]);
    test(net, "./demo/xor/data/10.png", "./demo/xor/data/10.txt");
    printf("xor: [1, 0], test: %f\n", l->input[0]);
    test(net, "./demo/xor/data/01.png", "./demo/xor/data/01.txt");
    printf("xor: [0, 1], test: %f\n", l->input[0]);
}

void train_xor_demo(char **argv)
{
    Network *net = load_network("./cfg/xor.cfg");
    init_network(net, "./demo/xor/xor.data", argv[3]);
    int x = strtol(argv[2], NULL, 10);
    printf("%d\n",x);
    train(net, x);
}

int main(int argc, char **argv)
{
    if (0 == strcmp(argv[1], "train")){
        train_xor_demo(argv);
    } else if (0 == strcmp(argv[1], "test")){
        test_xor_demo(argv);
    }
    return 0;
}
