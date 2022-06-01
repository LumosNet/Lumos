#include "manager.h"

void init_graph(Session sess, Graph g);

void init_avgpool_layer(Session sess, Layer l);
void init_connect_layer(Session sess, Layer l);
void init_convolutional_layer(Session sess, Layer l);
void init_im2col_layer(Session sess, Layer l);
void init_maxpool_layer(Session sess, Layer l);
void init_mse_layer(Session sess, Layer l);

void restore_graph(Session sess, Layer g);

void restore_avgpool_layer(Session sess, Layer l);
void restore_connect_layer(Session sess, Layer l);
void restore_convolutional_layer(Session sess, Layer l);
void restore_im2col_layer(Session sess, Layer l);
void restore_maxpool_layer(Session sess, Layer l);
void restore_mse_layer(Session sess, Layer l);