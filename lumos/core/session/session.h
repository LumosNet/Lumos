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

typedef struct session{
    int *command;
    Graph *graph;
    int batch;
    int subdivision;
    int epoch;
    float learning_rate;
    char *data_paths;
    char *truth_paths;

    int coretype;
} Session;

Session *create_session();
void set_batch(Session *sess, int batch);
void set_subdivision(Session *sess, int subdivision);
void set_epoch(Session *sess, int epoch);
void set_learning_rate(Session *sess, float learning_rate);
void set_data_paths(Session *sess, char *data_paths);
void set_truth_paths(Session *sess, char *truth_paths);

void session_execute_command(Session *sess, char *command);

#ifdef __cplusplus
}
#endif

#endif