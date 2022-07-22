#ifndef DISPATCH_H
#define DISPATCH_H

#include "session.h"
#include "manager.h"
#include "graph.h"
#include "layer.h"

#include "progress_bar.h"

#ifdef __cplusplus
extern "C" {
#endif

void session_train(Session *sess, float learning_rate);
void session_test(Session *sess, ProcessTestInformation process_test_information);

void forward_session(Session *sess);
void backward_session(Session *sess);
void update_session(Session *sess);

void create_train_scene(Session *sess, int h, int w, int c, int label_num, int truth_num, Label2Truth func, char *dataset_list_file, char *label_list_file);
void init_train_scene(Session *sess, int epoch, int batch, int subdivision, char *weights_file);

void create_test_scene(Session *sess, int h, int w, int c, int label_num, int truth_num, Label2Truth func, char *dataset_list_file, char *label_list_file);
void init_test_scene(Session *sess, char *weights_file);

#ifdef __cplusplus
}
#endif

#endif