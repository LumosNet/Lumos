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

int main(int argc, char **argv)
{
    Network *net = load_network("./cfg/lumos.cfg");
    init_network(net, "./mnist/mnist.data", NULL);
    train(net);
    // int offset = 0;
    // while (1){
    //     for (int i = 0; i < net->batch; ++i){
    //         int index = offset + i;
    //         if (index >= net->num) index -= net->num;
    //         printf("%s\n", net->label[index]);
    //         net->labels[i] = get_labels(net->label[index])[0];
    //     }
    //     offset += net->batch;
    //     if (offset >= net->num) offset -= net->num;
    // }
    return 0;
}