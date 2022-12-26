#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tsession.h"

int main(void)
{
    run_benchmarks("./lumos_t/benchmark/core/ops/pooling/avgpool.json");
    run_benchmarks("./lumos_t/benchmark/core/ops/pooling/avgpool_gradient.json");
    run_benchmarks("./lumos_t/benchmark/core/ops/pooling/maxpool.json");
    run_benchmarks("./lumos_t/benchmark/core/ops/pooling/maxpool_gradient.json");
    return 0;
}
