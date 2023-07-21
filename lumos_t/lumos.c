#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "run_test.h"

#define VERSION "0.1"

// void lumos(int argc, char **argv)
// {
//     if (argc <= 1){
//         fprintf(stderr, "Lumos fatal: No input option; use option --help for more information");
//         return;
//     }
//     if (0 == strcmp(argv[1], "--version") || 0 == strcmp(argv[1], "-v"))
//     {
//         char version[] = VERSION;
//         fprintf(stderr, "Lumos version: v%s\n", version);
//         fprintf(stderr, "This is free software; see the source for copying conditions.  There is NO\n");
//         fprintf(stderr, "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n");
//     }
//     else if (0 == strcmp(argv[1], "--runall") || 0 == strcmp(argv[1], "-ra"))
//     {
//         if (argc <= 2){
//             fprintf(stderr, "Lumos fatal: You miss the param of Testing module; use option --help for more information\n");
//             return ;
//         }
//         if (0 == strcmp(argv[2], "all")){
//             run_all_benchmarks("./lumos_t/benchmark/benchmarks.txt", 0);
//             run_all_benchmarks("./lumos_t/benchmark/benchmarks.txt", 1);
//         } else if (0 == strcmp(argv[2], "cpu")){
//             run_all_benchmarks("./lumos_t/benchmark/benchmarks.txt", 0);
//         } else if (0 == strcmp(argv[2], "gpu")){
//             run_all_benchmarks("./lumos_t/benchmark/benchmarks.txt", 1);
//         }
//     }
//     else if (0 == strcmp(argv[1], "--help") || 0 == strcmp(argv[1], "-h"))
//     {
//         fprintf(stderr, "Usage commands:\n");
//         fprintf(stderr, "    --version or -v : To get version\n");
//         fprintf(stderr, "    --path or -p : View The installation path\n");
//         fprintf(stderr, "    --runall or -ra : To run all the tests\n");
//         fprintf(stderr, "      runall all : To run all the tests\n");
//         fprintf(stderr, "      runall cpu : To run all the cpu tests\n");
//         fprintf(stderr, "      runall gpu : To run all the gpu tests\n");
//         fprintf(stderr, "Thank you for using Lumos deeplearning framework.\n");
//     }
// }

int main(int argc, char **argv)
{
    run_by_benchmark_file("./benchmark/core/ops/bias/add_bias.json", CPU);
    // lumos(argc, argv);
    return 0;
}
