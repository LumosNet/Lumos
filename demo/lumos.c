#include "alexnet_flower.h"

int main()
{
    // alexnet("gpu", "./backup/LW_f");
    // alexnet_detect("gpu", "./backup/LW_f");

    // xor("gpu", NULL);
    // xor_detect("gpu", "./backup/LW_f");

    // lenet5("gpu", NULL);
    // lenet5_detect("gpu", "./backup/LW_f");

    // lenet5_fmnist("gpu", NULL);

    alexnet_flower("gpu", NULL);
    return 0;
}