#include "lenet.h"

void lenet_label2truth(char **label, float *truth)
{
    int x = atoi(label[0]);
    one_hot_encoding(1, x, truth);
}

void lenet_process_test_information(char **label, float *truth, float *predict, float loss, char *data_path)
{
    fprintf(stderr, "Test Data Path: %s\n", data_path);
    fprintf(stderr, "Label:   %s\n", label[0]);
    fprintf(stderr, "Truth:   %f\n", truth[0]);
    fprintf(stderr, "Predict: %f\n", predict[0]);
    fprintf(stderr, "Loss:    %f\n\n", loss);
}

void lenet() {
    Graph *graph = create_graph("Lumos", 5);
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
    init_train_scene(sess, 4000, 16, 16, NULL);
    session_train(sess, 0.1, "/home/lumos/lumos.w");

    Session *t_sess = create_session();
    bind_graph(t_sess, graph);
    create_test_scene(t_sess, 32, 32, 1, 1, 10, lenet_label2truth, "/usr/local/lumos/data/mnist/test.txt", "/usr/local/lumos/data/mnist/test_label.txt");
    init_test_scene(t_sess, "/home/lumos/lumos.w");
    session_test(t_sess, lenet_process_test_information);
}
