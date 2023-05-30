#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tsession.h"

#define VERSION "0.1"

void lumos(int argc, char **argv)
{
    if (argc <= 1){
        fprintf(stderr, "Lumos fatal: No input option; use option --help for more information");
        return;
    }
    if (0 == strcmp(argv[1], "--version") || 0 == strcmp(argv[1], "-v"))
    {
        char version[] = VERSION;
        fprintf(stderr, "Lumos version: v%s\n", version);
        fprintf(stderr, "This is free software; see the source for copying conditions.  There is NO\n");
        fprintf(stderr, "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n");
    }
    else if (0 == strcmp(argv[1], "--runall") || 0 == strcmp(argv[1], "-ra"))
    {
        if (argc <= 2){
            fprintf(stderr, "Lumos fatal: You miss the param of Testing module; use option --help for more information\n");
            return ;
        }
        if (0 == strcmp(argv[2], "all")){
            run_all_benchmarks("./lumos_t/benchmark/benchmarks.txt", 0);
            run_all_benchmarks("./lumos_t/benchmark/benchmarks.txt", 1);
        } else if (0 == strcmp(argv[2], "cpu")){
            run_all_benchmarks("./lumos_t/benchmark/benchmarks.txt", 0);
        } else if (0 == strcmp(argv[2], "gpu")){
            run_all_benchmarks("./lumos_t/benchmark/benchmarks.txt", 1);
        }
    }
}

int main(int argc, char **argv)
{
    lumos(argc, argv);
    return 0;
}
