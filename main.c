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

void lenet_label2truth(char **label, float *truth)
{
    int x = atoi(label[0]);
    one_hot_encoding(2, x, truth);
}

void lenet_process_test_information(char **label, float *truth, float *predict, float loss, char *data_path)
{
    fprintf(stderr, "Test Data Path: %s\n", data_path);
    fprintf(stderr, "Label:   %s\n", label[0]);
    fprintf(stderr, "Truth:   %f\n", truth[0]);
    fprintf(stderr, "Predict: %f\n", predict[0]);
    fprintf(stderr, "Loss:    %f\n\n", loss);
}

void test() {
    Graph *graph = create_graph("Lumos", 3);
    Layer *l1 = make_convolutional_layer(1, 3, 1, 0, 1, 0, "logistic");
    Layer *l2 = make_im2col_layer(1);
    Layer *l3 = make_connect_layer(2, 1, "logistic");
    Layer *l4 = make_mse_layer(2);
    append_layer2grpah(graph, l1);
    append_layer2grpah(graph, l2);
    append_layer2grpah(graph, l3);
    append_layer2grpah(graph, l4);

    Session *sess = create_session();
    bind_graph(sess, graph);
    create_train_scene(sess, 4, 4, 1, 1, 2, lenet_label2truth, "./data/train.txt", "./data/label.txt");
    init_train_scene(sess, 1, 1, 1, NULL);
    session_train(sess, 1, "./lumos.w");
}

int main()
{
    test();
    return 0;
}
