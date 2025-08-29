#include "lenet5_mnist.h"
#include "xor.h"

int main()
{
    lenet5_mnist("cpu", NULL);
    // lenet5_mnist_detect("gpu", "./backup/LW_f");

    // xor("cpu", NULL);
    // xor_detect("gpu", "./backup/LW_f");
    return 0;
}