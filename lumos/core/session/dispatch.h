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

void session_train(Session *sess, char *weights_path);
void session_test(Session *sess);

void forward_session(Session *sess);
void backward_session(Session *sess);

void init_train_scene(Session *sess, char *weights_file);
void init_test_scene(Session *sess, char *weights_file);

void set_run_type(Session *sess, int train);

void test_information(float *truth, float *predict, float loss, char *data_path);

#ifdef __cplusplus
}
#endif

#endif