#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"
#include "array.h"
#include "victor.h"

#include "loss.h"

int main(int argc, char **argv)
{
    float list1[] = {1, 2, 3};
    float list2[] = {-1, -2, -3};
    Victor *v1 = victor_list(3, 1, list1);
    Victor *v2 = victor_list(3, 1, list2);
    float res = mse(v1, v2);
    printf("%f\n", res);
    return 0;
}