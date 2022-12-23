#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include "bias_gpu.h"
#include "cpu_gpu.h"
#include "tsession.h"

int main(void)
{
    run_benchmarks("./lumos_t/benchmark/core_cu/ops/im2col/im2col_gpu.json");
    return 0;
}
