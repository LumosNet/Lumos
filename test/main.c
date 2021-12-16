#include "lumos.h"
#include "network.h"
#include "utils.h"
#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    Network *net = load_network("./cfg/lumos.cfg");
    init_network(net, "./mnist/mnist.data", NULL);

    load_train_data(net, 0);

    Layer l = net->layers[0];
    l.input = net->output;
    l.forward(l, net[0]);
    // net->output = l.output;

    // for (int i = 0; i < net->n; ++i){
    //     Layer l = net->layers[i];
    //     l.input = net->output;
    //     printf("start\n");
    //     l.forward(l, net[0]);
    //     net->output = l.output;
    // }

    return 0;
}