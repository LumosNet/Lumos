#ifndef LENET5_MNIST_H
#define LENET5_MNIST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "connect_layer.h"
#include "convolutional_layer.h"
#include "avgpool_layer.h"
#include "im2col_layer.h"
#include "softmax_layer.h"
#include "mse_layer.h"
#include "graph.h"
#include "layer.h"
#include "session.h"

#ifdef __cplusplus
extern "C" {
#endif

void lenet5_mnist(char *type, char *path);
void lenet5_mnist_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif
