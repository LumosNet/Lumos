#ifndef SESSION_H
#define SESSION_H

#include <stdio.h>
#include <stdlib.h>

#include "graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct session{
    Graph *graph;

    int input_w;
    int input_h;
    int input_c;

    int workspace_size;

    float *workspace;
    float *output;
    float *layer_delta;
    float *net_delta;
} session, Session;

void session_bind_graph(Session sess, Graph graph);
void session_unbind_graph(Session sess);
void session_bind_data(Session sess);
void session_unbind_data(Session sess);
void session_run(Session sess);
void session_del(Session sess);

#ifdef __cplusplus
}
#endif

#endif