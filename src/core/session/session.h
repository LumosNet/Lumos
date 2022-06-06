#ifndef SESSION_H
#define SESSION_H

#include <stdio.h>
#include <stdlib.h>

#include "graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct session{
    Graph graph;

    int epoch;
    int batch;
    int subdivision;

    int width;
    int height;
    int channel;

    float learning_rate;
    size_t workspace_size;

    float *workspace;
    float *input;
    float *output;
    float *layer_delta;
    float *net_delta;

    float *weights;
} session, Session;

/*
    每次读取一个subdivision的数据
*/

void session_bind_graph(Session sess, Graph graph);
void session_unbind_graph(Session sess);
void session_bind_data(Session sess);
void session_unbind_data(Session sess);
void session_run(Session sess);
void session_del(Session sess);

// 从index读取num个数据
void load_data(Session sess, int index, int num);

void save_weigths(Session sess, char *path);
void load_weights(Session sess, char *path);

#ifdef __cplusplus
}
#endif

#endif