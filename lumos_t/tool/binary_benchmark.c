#include "binary_benchmark.h"

void *get_binary_benchmark(FILE *fp)
{
    int head[1];
    fread(head, sizeof(int), 1, fp);
    void *buffer = malloc(head[0]);
    fread(buffer, 1, head[0], fp);
    return buffer;
}

int analysis_interface(void *buffer, int offset); //返回接口标识
int analysis_usecase(void *buffer, int offset); //返回用例参数个数
void analysis_parameters(void *buffer, int offset, void **parameters);