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
#include "optimize.h"
#include "binary_f.h"

#ifdef __cplusplus
extern "C" {
#endif

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

    char *weights_path;

    LrScheduler *lrscheduler;
} Session;

Session *create_session(Graph *graph, int h, int w, int c, int truth_num, char *type, char *path);
void init_session(Session *sess, char *data_path, char *label_path);

void bind_train_data(Session *sess, char *path);
void bind_train_label(Session *sess, char *path);

void set_train_params(Session *sess, int epoch, int batch, int subdivision, float learning_rate);
void set_detect_params(Session *sess);
void create_workspace(Session *sess);
void train(Session *sess, int binary);
void detect_classification(Session *sess, int binary);

void load_train_data(Session *sess, int index);
void load_train_data_binary(Session *sess, int index);
void load_train_label(Session *sess, int index);

void lr_scheduler_step(Session *sess, int step_size, float gamma);
void lr_scheduler_multistep(Session *sess, int *milestones, int num, float gamma);
void lr_scheduler_exponential(Session *sess, float gamma);
void lr_scheduler_cosineannealing(Session *sess, int T_max, float lr_min);

void init_constant(Layer *l, float x);
void init_normal(Layer *l, float mean, float std);
void init_uniform(Layer *l, float min, float max);
void init_kaiming_normal(Layer *l, float a, char *mode, char *nonlinearity);
void init_kaiming_uniform(Layer *l, float a, char *mode, char *nonlinearity);

#ifdef __cplusplus
}
#endif

#endif