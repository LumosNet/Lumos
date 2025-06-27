#include "lenet5_cifar.h"
#include "alexnet.h"
#include "xor.h"
#include "binary_f.h"
#include "lenet5.h"

int main()
{
    alexnet("gpu", NULL);
    alexnet_detect("gpu", "./backup/LW_f");

    // xor("gpu", NULL);
    // xor_detect("gpu", "./backup/LW_f");

    // lenet5("gpu", NULL);
    // lenet5_detect("gpu", "./backup/LW_f");
    return 0;
}