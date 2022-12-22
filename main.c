#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tsession.h"

int main(void)
{
    run_benchmarks("./lumos_t/benchmark/ops/im2col/col2im.json");
    return 0;
}
