#include "lenet5_cifar.h"
#include "alexnet.h"
#include "xor.h"
#include "binary_f.h"
#include "lenet5.h"

int main()
{
    // alexnet("gpu", "./build/LW_f");
    // alexnet_detect("gpu", "./build/LW_f");

    xor("gpu", NULL);
    xor_detect("gpu", "./build/LW_f");
    return 0;
}