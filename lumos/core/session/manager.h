#ifndef MANAGER_H
#define MANAGER_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include "cJSON.h"
#include "cJSON_Utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "session.h"
#include "graph.h"
#include "layer.h"
#include "weights_init.h"

#ifdef __cplusplus
extern "C" {
#endif

void create_run_memory(Session *sess);
void create_workspace_memory(Session *sess);
void create_input_memory(Session *sess);
void create_output_memory(Session *sess);
void create_weights_memory(Session *sess);
void create_delta_memory(Session *sess);
void create_label_memory(Session *sess);
void create_loss_memory(Session *sess);
void create_truth_memory(Session *sess);
void create_predicts_memory(Session *sess);
void create_maxpool_index_memory(Session *sess);
void create_dropout_rand_memory(Session *sess);

void set_graph_memory(Session *sess);
void set_graph_weight(Session *sess);
void set_label(Session *sess);
void set_loss_memory(Session *sess);
void set_truth_memory(Session *sess);
void set_maxpool_index_memory(Session *sess);
void set_dropout_rand_memory(Session *sess);

void get_workspace_size(Session *sess);
void statistics_memory_occupy_size(Session *sess);

void init_weights(Session *sess, char *weights_file);

// 从index读取num个数据
void load_train_data(Session *sess, int index, int num);
void load_train_label(Session *sess, int index, int num);

void load_test_data(Session *sess, int index);
void load_test_label(Session *sess, int index);

void save_weigths(Session *sess, char *path);
void load_weights(Session *sess, char *path);

Session *load_session_json(char *graph_path, char *coretype);
Initializer load_initializer_json(cJSON *cjson_init);
Graph *load_graph_json(cJSON *cjson_graph);

#ifdef __cplusplus
}
#endif

#endif