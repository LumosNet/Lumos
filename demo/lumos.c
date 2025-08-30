#include "lenet5_cifar10.h"
#include "lenet5_mnist.h"

int main()
{
    lenet5_cifar10("gpu", NULL);
    lenet5_cifar10_detect("gpu", "./LuWeights");
    // lenet5_mnist("gpu", NULL);
    // lenet5_mnist_detect("gpu", "./LuWeights");
}