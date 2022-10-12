#include <stdio.h>
#include <stdlib.h>

#include "cpu.h"
#include "graph.h"
#include "layer.h"
#include "im2col_layer.h"
#include "connect_layer.h"
#include "convolutional_layer.h"
#include "avgpool_layer.h"
#include "mse_layer.h"
#include "session.h"
#include "manager.h"
#include "dispatch.h"
#include "utest.h"

void lenet_label2truth(char **label, float *truth)
{
    int x = atoi(label[0]);
    one_hot_encoding(10, x, truth);
}

void test_forward_session(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    Layer *l;
    float *input = sess->input;
    for (int i = 0; i < graph->layer_num; ++i)
    {
        l = layers[i];
        l->input = input;
        l->forward(*l, sess->subdivision);
        input = l->output;
    }
}

void test_backward_session(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    Layer *l;
    float rate = -sess->learning_rate / (float)sess->batch;
    float *delta = NULL;
    for (int i = graph->layer_num - 1; i >= 0; --i)
    {
        l = layers[i];
        l->backward(*l, rate, sess->subdivision, delta);
        delta = l->delta;
    }
}

void test_data_flow() {
    test_run("test_data_flow");
    Graph *graph = create_graph("Lumos", 9);
    Layer *l1 = make_convolutional_layer(6, 5, 1, 0, 1, 1, "logistic");
    Layer *l2 = make_avgpool_layer(2);
    Layer *l3 = make_convolutional_layer(16, 5, 1, 0, 1, 1, "logistic");
    Layer *l4 = make_avgpool_layer(2);
    Layer *l5 = make_convolutional_layer(120, 5, 1, 0, 1, 1, "logistic");
    Layer *l6 = make_im2col_layer(1);
    Layer *l7 = make_connect_layer(84, 1, "logistic");
    Layer *l8 = make_connect_layer(10, 1, "logistic");
    Layer *l9 = make_mse_layer(10);
    append_layer2grpah(graph, l1);
    append_layer2grpah(graph, l2);
    append_layer2grpah(graph, l3);
    append_layer2grpah(graph, l4);
    append_layer2grpah(graph, l5);
    append_layer2grpah(graph, l6);
    append_layer2grpah(graph, l7);
    append_layer2grpah(graph, l8);
    append_layer2grpah(graph, l9);

    Session *sess = create_session();
    bind_graph(sess, graph);
    create_train_scene(sess, 32, 32, 1, 1, 10, lenet_label2truth, "/usr/local/lumos/data/mnist/train.txt", "/usr/local/lumos/data/mnist/train_label.txt");
    init_train_scene(sess, 500, 2, 2, NULL);
    float learning_rate = 0.1;
    sess->learning_rate = learning_rate;
    for (int i = 0; i < sess->epoch; ++i)
    {
        int sub_epochs = (int)(sess->train_data_num / sess->batch);
        int sub_batchs = (int)(sess->batch / sess->subdivision);
        for (int j = 0; j < sub_epochs; ++j)
        {
            for (int k = 0; k < sub_batchs; ++k)
            {
                load_train_data(sess, j * sess->batch + k * sess->subdivision, sess->subdivision);
                load_train_label(sess, j * sess->batch + k * sess->subdivision, sess->subdivision);
                test_forward_session(sess);
                test_backward_session(sess);
            }
            memcpy(sess->weights, sess->update_weights, sess->weights_size * sizeof(float));
        }
    }
}

int main()
{
    lenet();
    return 0;
}
