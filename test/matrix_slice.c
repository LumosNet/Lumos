#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "list.h"

// void slice(tensor *m, int **sections, float *workspace);

void test()
{
    int size[] = {4,4,4};
    float list[] = {\
     0,  1,  2,  3, \
     4,  5,  6,  7, \
     8,  9, 10, 11, \
    12, 13, 14, 15, \
    \
    16, 17, 18, 19, \
    20, 21, 22, 23, \
    24, 25, 26, 27, \
    18, 29, 30, 31, \
    \
    32, 33, 34, 35, \
    36, 37, 38, 39, \
    40, 41, 42, 43, \
    44, 45, 46, 47, \
    \
    48, 49, 50, 51, \
    52, 53, 54, 55, \
    56, 57, 58, 59, \
    60, 61, 62, 63  \
    };
    tensor *m = list_to_tensor(3, size, list);
    int **sections = malloc(3*sizeof(int*));
    int section_1[] = {1, 0, 4};
    int section_2[] = {2, 0, 1, 2, 4};
    int section_3[] = {2, 2, 3, 3, 4};
    sections[0] = section_1;
    sections[1] = section_2;
    sections[2] = section_3;
    float *workspace = malloc(24*sizeof(float));
    int offset = multing_int_list(m->size, 0, m->dim);
    printf("offset: %d\n", offset);
    __slice(m, sections, workspace, 2);
    for (int i = 0; i < 24; ++i){
        printf("%f\n", workspace[i]);
    }
}

int main(int argc, char **argv)
{
    test();
    return 0;
}