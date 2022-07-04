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
    Graph *graph;

    int epoch;
    int batch;
    int subdivision;

    int width;
    int height;
    int channel;

    float *loss;

    int label_num;

    float learning_rate;
    size_t workspace_size;
    size_t weights_size;

    float *workspace;
    float *input;
    float *output;
    float *layer_delta;
    char **label;

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
} Session;

/*
    每次读取一个subdivision的数据
*/

/*
    06.14：session创建。
           init：训练超参、数据维度大小
           绑定图
           绑定数据集
           初始化图：创建内存，分配内存，初始化权重
           运行
*/

Session *create_session();
void del_session();

void bind_graph(Session *sess, Graph *graph);
void bind_train_data(Session *sess, char *path);
void bind_test_data(Session *sess, char *path);
void bind_label(Session *sess, int label_num, char *path);

void set_input_dimension(Session *sess, int h, int w, int c);
void set_train_params(Session *sess, int epoch, int batch, int subdivision, float learning_rate);

#ifdef __cplusplus
}
#endif

#endif