#include "binary_benchmark.h"

void *get_binary_benchmark(char *path)
{
    int head[1];
    FILE *fp = fopen(path, "rb");
    fread(head, sizeof(int), 1, fp);
    printf("%ld \n", sizeof(int));
    printf("%d \n", head[0]);
    void *buffer = malloc(head[0]);
    fread(buffer, 1, head[0], fp);
    fclose(fp);
    return buffer;
}
