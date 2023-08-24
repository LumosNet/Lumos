#ifndef GRAPH_H
#define GRAPH_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct graph{
    int num;
    Layer *layers;
} graph, Graph;

#ifdef __cplusplus
}
#endif

#endif