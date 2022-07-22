#ifndef MANAGER_H
#define MANAGER_H

#include <stdio.h>
#include <stdlib.h>

#include "session.h"
#include "graph.h"
#include "layer.h"

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
void create_maxpool_index_memory(Session *sess);

void set_graph_memory(Session *sess);
void set_graph_weight(Session *sess);
void set_label(Session *sess);
void set_loss_memory(Session *sess);
void set_truth_memory(Session *sess);
void set_maxpool_index_memory(Session *sess);

void get_workspace_size(Session *sess);
void statistics_memory_occupy_size(Session *sess);

void init_weights(Session *sess, char *weights_file);

// 从index读取num个数据
void load_train_data(Session *sess, int index, int num);
void load_train_label(Session *sess, int index, int num);

void load_test_data(Session *sess, int index);
char **load_test_label(Session *sess, int index);

void save_weigths(Session *sess, char *path);
void load_weights(Session *sess, char *path);

#ifdef __cplusplus
}
#endif

#endif