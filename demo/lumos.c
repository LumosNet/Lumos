#include "lenet5_cifar10.h"
#include "lenet5_mnist.h"
#include "xor.h"
#include "alexnet_xray.h"

int main()
{
    // lenet5_cifar10("gpu", NULL);
    // lenet5_cifar10_detect("gpu", "./LuWeights");
    // lenet5_mnist("gpu", NULL);
    // lenet5_mnist_detect("gpu", "./LuWeights");
    // xor("cpu", NULL);
    // xor_detect("cpu", "./LuWeights");
    alexnet_xray("gpu", NULL);
    alexnet_xray_detect("gpu", "./LuWeights");
}