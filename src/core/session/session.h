#ifndef SESSION_H
#define SESSION_H

#include <stdio.h>
#include <stdlib.h>

#include "graph.h"
#include "text_f.h"
#include "binary_f.h"
#include "image.h"

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

    float *weights;
    float *update_weights;

    int train_data_num;
    char **train_data_paths;
} session, Session;

/*
    每次读取一个subdivision的数据
*/

Session create_session();
void del_session();

void bind_graph(Session sess, Graph graph);
void bind_train_data(Session sess, char *path);
void bind_test_data(Session sess, char *path);

void set_input_dimension(Session sess, int h, int w, int c);
void set_train_params(Session sess, int epoch, int batch, int subdivision, float learning_rate);

// 从index读取num个数据
void load_data(Session sess, int index, int num);

void save_weigths(Session sess, char *path);
void load_weights(Session sess, char *path);

#ifdef __cplusplus
}
#endif

#endif