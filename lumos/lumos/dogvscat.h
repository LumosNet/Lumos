#ifndef DOGVSCAT_H
#define DOGVSCAT_H

#include <stdio.h>
#include <stdlib.h>

#include "cpu.h"
#include "graph.h"
#include "layer.h"
#include "im2col_layer.h"
#include "connect_layer.h"
#include "convolutional_layer.h"
#include "avgpool_layer.h"
#include "mse_layer.h"
#include "session.h"
#include "manager.h"
#include "dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

void dogvscat();

#ifdef __cplusplus
}
#endif

#endif