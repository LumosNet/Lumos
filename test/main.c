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
#include <string.h>

int main(int argc, char **argv)
{
    Network *net = load_network(argv[1]);
    return 0;
}