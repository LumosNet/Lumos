#include "run_test.h"

int main(int argc, char **argv)
{
    // TestInterface FUNC = call_layer_delta;
    FILE *logfp = fopen("./log/logging", "w");
    // run_by_benchmark_file("./lumos_t/benchmark/memory/layer_delta.json", FUNC, CPU, logfp);
    run_all_benchmark(GPU, logfp);
    return 0;
}