#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <stdlib.h>

#include "cpu.h"
#include "graph.h"
#include "layer.h"
#include "im2col_layer.h"
#include "connect_layer.h"
#include "mse_layer.h"
#include "session.h"
#include "manager.h"
#include "dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

void full_connect_mnist();

#ifdef __cplusplus
}
#endif

#endif