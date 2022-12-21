#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tsession.h"

int main(void)
{
    run_benchmarks("./lumos_t/benchmark/ops/gemm/gemm.json");
    return 0;
}
