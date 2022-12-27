#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tsession.h"

int main(void)
{
    // run_benchmarks("./lumos_t/benchmark/core_cu/ops/im2col/im2col_gpu.json");
    run_benchmarks("./lumos_t/benchmark/core_cu/ops/im2col/col2im_gpu.json");
    return 0;
}
