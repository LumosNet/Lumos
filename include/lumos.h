#ifndef LUMOS_H
#define LUMOS_H

#include <stdio.h>
#include <stdlib.h>

#define CPU 0
#define GPU 1

typedef struct session Session;
typedef struct graph Graph;
typedef struct layer Layer;

Session *create_session(Graph *graph, int h, int w, int c, int truth_num, char *type, char *path);
void init_session(Session *sess, char *data_path, char *label_path);
void set_train_params(Session *sess, int epoch, int batch, int subdivision, float learning_rate);
void set_detect_params(Session *sess);
void train(Session *sess);
void detect_classification(Session *sess);

Graph *create_graph();
void append_layer2grpah(Graph *graph, Layer *l);

Layer *make_avgpool_layer(int ksize, int stride, int pad);
Layer *make_connect_layer(int output, int bias, char *active);
Layer *make_convolutional_layer(int filters, int ksize, int stride, int pad, int bias, char *active);
Layer *make_dropout_layer(float probability);
Layer *make_im2col_layer();
Layer *make_maxpool_layer(int ksize, int stride, int pad);
Layer *make_softmax_layer(int group);

Layer *make_mse_layer(int group);

void init_constant(Layer *l, float x);
void init_normal(Layer *l, float mean, float std);
void init_uniform(Layer *l, float min, float max);
void init_kaiming_normal(Layer *l, float a, char *mode, char *nonlinearity);
void init_kaiming_uniform(Layer *l, float a, char *mode, char *nonlinearity);

#endif