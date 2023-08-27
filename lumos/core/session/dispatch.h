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

void train(Session *sess, char *weights_path);
void detect(Session *sess);

void forward(Session *sess);
void backward(Session *sess);

void init_train_scene(Session *sess, char *weights_file);
void init_test_scene(Session *sess, char *weights_file);

#ifdef __cplusplus
}
#endif

#endif