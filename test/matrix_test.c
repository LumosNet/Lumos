#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "AS.h"

void test_replace_mindex_to_lindex(int dim, int *size, int *index)
{
    int lindex = replace_mindex_to_lindex(dim, size, index);
    printf("Lindex: %d\n", lindex);
}

void test_replace_lindex_to_mindex(int dim, int *size, int index)
{
    int *mindex = replace_lindex_to_mindex(dim, size, index);
    printf("Mindex: ");
    for (int i = 0; i < dim; ++i){
        printf("%d ", mindex[i]);
    }
    printf("\n");
}

void test_mix_lindex_mindex()
{
    int dim = 3;
    int size[] = {4,4,4};
    int index = 0;
    printf("Index: %d\n", index);
    int *mindex = replace_lindex_to_mindex(dim, size, index);
    printf("Mindex: ");
    for (int i = 0; i < dim; ++i){
        printf("%d ", mindex[i]);
    }
    printf("\n");
    int lindex = replace_mindex_to_lindex(dim, size, mindex);
    printf("Lindex: %d\n", lindex);
}

int main(int argc, char **argv)
{
    test_mix_lindex_mindex();
}