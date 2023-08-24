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

typedef struct session{
    Graph *graph;
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

    float *workspace;
} Session;

void bind_train_data(Session *sess, char *path);
void bind_train_label(Session *sess, char *path);
void create_workspace(Session *sess);
int count_running_memsize(Session *sess);

#ifdef __cplusplus
}
#endif
#endif
