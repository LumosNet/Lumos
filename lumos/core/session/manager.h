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
#include "dataset.h"
#include "graph.h"
#include "layer.h"
#include "weights_init.h"

#ifdef __cplusplus
extern "C" {
#endif

Session *load_session_json(char *graph_path, char *coretype);
Initializer load_initializer_json(cJSON *cjson_init);
Graph *load_graph_json(cJSON *cjson_graph);

Layer **load_layers(cJSON *cjson_graph);
Layer *load_layer_json(cJSON *cjson_layer);

Layer *load_avgpool_layer_json(cJSON *cjson_layer);
Layer *load_connect_layer_json(cJSON *cjson_layer);
Layer *load_convolutional_layer_json(cJSON *cjson_layer);
Layer *load_dropout_layer_json(cJSON *cjson_layer);
Layer *load_im2col_layer_json(cJSON *cjson_layer);
Layer *load_maxpool_layer_json(cJSON *cjson_layer);
Layer *load_softmax_layer_json(cJSON *cjson_layer);
Layer *load_mse_layer_json(cJSON *cjson_layer);
Layer *load_shortcut_layer_json(cJSON *cjson_layer);

void train(Session *sess);
void detect(Session *sess);

void run_forward(Session *sess);
void run_backward(Session *sess);
void run_update(Session *sess);

void init_running_scene(Session *sess, char *weights_file);
void clean_running_scene(Session *sess);

#ifdef __cplusplus
}
#endif

#endif