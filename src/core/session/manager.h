#ifndef MANAGER_H
#define MANAGER_H

#include <stdio.h>
#include <stdlib.h>

#include "session.h"
#include "graph.h"
#include "layer.h"

#ifdef __cplusplus
extern "C" {
#endif

void create_run_memory(Session sess);
void create_workspace_memory(Session sess);
void create_output_memory(Session sess);
void create_delta_memory(Session sess);

void set_graph_memory();
void set_layer_output_memory();
void set_layer_weight_memory();

#ifdef __cplusplus
}
#endif

#endif