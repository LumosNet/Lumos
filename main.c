#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tsession.h"

int main(void)
{
    run_benchmarks("./lumos_t/benchmark/core_cu/ops/pooling/avgpool_gpu.json");
    run_benchmarks("./lumos_t/benchmark/core_cu/ops/pooling/avgpool_gradient_gpu.json");
    run_benchmarks("./lumos_t/benchmark/core_cu/ops/pooling/maxpool_gpu.json");
    run_benchmarks("./lumos_t/benchmark/core_cu/ops/pooling/maxpool_gradient_gpu.json");
    return 0;
}
