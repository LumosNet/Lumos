#include "image.h"
#include "gray_process.h"
#include "array.h"
#include "im2col.h"
#include "parser.h"
#include "utils.h"
#include "network.h"
#include "lumos.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    Network *net = load_network("./cfg/lumos.cfg");
    return 0;
}