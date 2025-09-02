#include "cifar10.h"
#include "lenet5_fmnist.h"
#include "xor.h"
#include "alexnet_xray.h"

int main()
{
    // cifar10("gpu", NULL);
    // cifar10_detect("gpu", "./LuWeights");
    lenet5_fmnist("gpu", "./LuWeights");
    lenet5_fmnist_detect("gpu", "./LuWeights");
    // xor("cpu", NULL);
    // xor_detect("cpu", "./LuWeights");
    // alexnet_xray("gpu", NULL);
    // alexnet_xray_detect("gpu", "./LuWeights");
}