#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avgpool_layer.h"
#include "connect_layer.h"
#include "convolutional_layer.h"
#include "im2col_layer.h"
#include "maxpool_layer.h"
#include "softmax_layer.h"
#include "mse_layer.h"
#include "graph.h"
#include "layer.h"
#include "session.h"
#include "gemm_gpu.h"
#include "cpu_gpu.h"

#include "analysis_benchmark_file.h"
#include "run_test.h"
#include "avgpool_layer_call.h"
#include "avgpool_layer_gpu_call.h"

#include "connect_layer_call.h"
#include "connect_layer_gpu_call.h"

#include "mse_layer_call.h"
#include "mse_layer_gpu_call.h"

#define VERSION "0.1"

void xor(char *type)
{
    Graph *g = create_graph();
    Layer *l1 = make_im2col_layer();
    Layer *l2 = make_connect_layer(4, 1, "relu");
    Layer *l3 = make_connect_layer(2, 1, "relu");
    Layer *l4 = make_mse_layer(2);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    Session *sess = create_session(g, 1, 2, 1, 2, type);
    set_train_params(sess, 50, 2, 2, 0.1);
    init_session(sess, "./data/xor/data.txt", "./data/xor/label.txt");
    train(sess);
}

void mnist(char *type)
{
    Graph *g = create_graph();
    Layer *l1 = make_im2col_layer();
    Layer *l2 = make_connect_layer(128, 1, "relu");
    Layer *l3 = make_connect_layer(64, 1, "relu");
    Layer *l4 = make_connect_layer(10, 1, "relu");
    Layer *l5 = make_mse_layer(10);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    Session *sess = create_session(g, 32, 32, 1, 10, type);
    set_train_params(sess, 20, 16, 16, 0.1);
    init_session(sess, "./data/mnist/train.txt", "./data/mnist/train_label.txt");
    train(sess);
}

void lenet5(char *type)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(6, 5, 1, 0, 1, "relu");
    Layer *l2 = make_avgpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(16, 5, 1, 0, 1, "relu");
    Layer *l4 = make_avgpool_layer(2, 2, 0);
    Layer *l5 = make_convolutional_layer(120, 5, 1, 0, 1, "relu");
    Layer *l6 = make_im2col_layer();
    Layer *l7 = make_connect_layer(84, 1, "relu");
    Layer *l8 = make_connect_layer(10, 1, "relu");
    // Layer *l9 = make_softmax_layer(10);
    Layer *l10 = make_mse_layer(10);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);
    // append_layer2grpah(g, l9);
    append_layer2grpah(g, l10);
    Session *sess = create_session(g, 32, 32, 1, 10, type);
    set_train_params(sess, 15, 16, 8, 0.1);
    init_session(sess, "./data/mnist/train.txt", "./data/mnist/train_label.txt");
    train(sess);
}

int main(int argc, char **argv)
{
    TestInterface FUNC = call_forward_mse_layer;
    run_by_benchmark_file("./lumos_t/benchmark/core/graph/loss_layer/mse_layer/forward_mse_layer.json", FUNC, CPU);
    return 0;
}
