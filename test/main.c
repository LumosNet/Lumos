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


    for (int i = 0; i < net->n; ++i){
        Layer l = net->layers[i];
        l.input = net->output;
        printf("start\n");
        l.forward(l, net[0]);
        net->output = l.output;
    }

    // int *a = malloc(10*sizeof(int));
    // for (int i = 0; i < 10; ++i){
    //     a[i] = 0;
    //     a[i] = i+1;
    // }

    // int *b = a+2;
    // printf("%d %d\n", a[0], b[0]);
    printf("end\n");

    return 0;
}