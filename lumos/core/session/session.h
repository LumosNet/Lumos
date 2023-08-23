#ifndef SESSION_H
#define SESSION_H

#include <stdio.h>
#include <stdlib.h>

#include "graph.h"
#include "text_f.h"
#include "binary_f.h"
#include "image.h"
#include "weights_init.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CPU 0
#define GPU 1

typedef void (*label2truth) (char **, float *);
typedef label2truth Label2Truth;

typedef void (*process_test_information) (char **, float *, float *, float, char *);
typedef process_test_information ProcessTestInformation;

// typedef struct session{
//     Graph *graph;

//     int coretype;
//     int epoch;
//     int batch;
//     int subdivision;

//     int width;
//     int height;
//     int channel;

//     float *loss;

//     float learning_rate;
//     size_t workspace_size;

//     float *workspace;
//     float *input;
//     float *output;
//     float *delta;

//     char **label;
//     int label_num;

//     float *truth;
//     int truth_num;

//     float *predicts;

//     int train_data_num;
//     char **train_data_paths;
//     char **train_label_paths;

//     int test_data_num;
//     char **test_data_paths;
//     char **test_label_paths;

//     int memory_size;

//     float *x_norm;
//     float *mean;
//     float *variance;
//     float *roll_mean;
//     float *roll_variance;
//     float *normalize_x;

//     int x_norm_size;
//     int variance_size;
//     int normalize_x_size;

//     Label2Truth label2truth;
//     Initializer w_init;
// } Session;

typedef struct session{
    int batch;
    int subdivison;

    int height;
    int width;
    int channel;

    int dataset_num;
    char **dataset_pathes;
    char **labelset_pathes;

    int index;
    float **input;
    float **truth;
} Session;

Session *create_session(char *type, Initializer w_init);

void bind_graph(Session *sess, Graph *graph);
void bind_train_data(Session *sess, char *path);
void bind_test_data(Session *sess, char *path);
void bind_train_label(Session *sess, char *path);
void bind_test_label(Session *sess, char *path);
void bind_label2truth_func(Session *sess, int truth_num, Label2Truth func);

void set_input_dimension(Session *sess, int h, int w, int c);
void set_train_params(Session *sess, int epoch, int batch, int subdivision, float learning_rate);

#ifdef __cplusplus
}
#endif

#endif