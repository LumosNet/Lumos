#ifndef SESSION_H
#define SESSION_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "graph.h"
#include "text_f.h"
#include "binary_f.h"
#include "image.h"
#include "progress_bar.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CPU 0
#define GPU 1

typedef struct session{
    Graph *graph;

    int coretype;
    int epoch;
    int batch;
    int subdivision;

    int width;
    int height;
    int channel;

    float *loss;

    float learning_rate;
    size_t workspace_size;

    float *workspace;
    float *input;

    float *truth;
    int truth_num;

    int train_data_num;
    char **train_data_paths;
    char **train_label_paths;

    int test_data_num;
    char **test_data_paths;
    char **test_label_paths;

    char *weights_path;
} Session;

Session *create_session(Graph *graph, int h, int w, int c, int truth_num, char *type, char *path);
void init_session(Session *sess, char *data_path, char *label_path);

void bind_train_data(Session *sess, char *path);
void bind_test_data(Session *sess, char *path);
void bind_train_label(Session *sess, char *path);
void bind_test_label(Session *sess, char *path);

void set_train_params(Session *sess, int epoch, int batch, int subdivision, float learning_rate);
void create_workspace(Session *sess);
void train(Session *sess);
void detect(Session *sess);

void load_train_data(Session *sess, int index);
void load_train_label(Session *sess, int index);

#ifdef __cplusplus
}
#endif

#endif