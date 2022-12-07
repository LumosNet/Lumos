#include "acall.h"

void test(char *path)
{
    FILE *fp = fopen(path, "rb");
    void *benchmark = get_binary_benchmark(fp);
    int interface_f = analysis_interface(benchmark, 2);
    int param_num = analysis_usecase(benchmark, 2);
    void **params = malloc(param_num*sizeof(void*));
    analysis_parameters(benchmark, 2, params);
    call();
}
