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

typedef void (*label2truth) (char **, float *);
typedef label2truth Label2Truth;

typedef void (*process_test_information) (char **, float *, float *, float, char *);
typedef process_test_information ProcessTestInformation;

typedef struct session{
    Graph *graph;

    int epoch;
    int batch;
    int subdivision;

    int width;
    int height;
    int channel;

    float *loss;

    float learning_rate;
    size_t workspace_size;
    size_t weights_size;

    float *workspace;
    float *input;
    float *output;
    float *layer_delta;

    char **label;
    int label_num;

    float *truth;
    int truth_num;

    float *predicts;

    int *maxpool_index;

    float *weights;
    float *update_weights;

    int train_data_num;
    char **train_data_paths;
    char **train_label_paths;

    int test_data_num;
    char **test_data_paths;
    char **test_label_paths;

    int memory_size;

    Label2Truth label2truth;

#ifdef GPU
    float *workspace_gpu;
    float *input_gpu;
    float *output_gpu;
    float *weights_gpu;
    float *update_weights_gpu;
    float *layer_delta_gpu;
    float *loss_gpu;
    float *truth_gpu;
    int *maxpool_index_gpu;
#endif

} Session;

Session *create_session();
void del_session();

void bind_graph(Session *sess, Graph *graph);
void bind_train_data(Session *sess, char *path);
void bind_test_data(Session *sess, char *path);
void bind_train_label(Session *sess, int label_num, char *path);
void bind_test_label(Session *sess, int label_num, char *path);
void bind_label2truth_func(Session *sess, int truth_num, Label2Truth func);

void set_input_dimension(Session *sess, int h, int w, int c);
void set_train_params(Session *sess, int epoch, int batch, int subdivision, float learning_rate);

#ifdef __cplusplus
}
#endif

#endif