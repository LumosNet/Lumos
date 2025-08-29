#include "run_test.h"
#include "im2col_call.h"
#include "pooling_call.h"
#include "layer_delta_call.h"

int main(int argc, char **argv)
{
    TestInterface FUNC = call_layer_delta;
    FILE *logfp = fopen("./log/logging", "w");
    run_by_benchmark_file("./lumos_t/benchmark/memory/layer_delta.json", FUNC, CPU, logfp);
    return 0;
}