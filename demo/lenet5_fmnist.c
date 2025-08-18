#include "lenet5_fmnist.h"

void lenet5_fmnist(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(6, 5, 1, 0, 1, 0, "logistic");
    Layer *l2 = make_maxpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(16, 5, 1, 0, 1, 0, "logistic");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_convolutional_layer(120, 5, 1, 0, 1, 0, "logistic");
    Layer *l6 = make_connect_layer(84, 1, "logistic");
    Layer *l7 = make_connect_layer(10, 1, "logistic");
    Layer *l8 = make_softmax_layer(10);
    Layer *l9 = make_mse_layer(10);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);
    append_layer2grpah(g, l9);
    Session *sess = create_session(g, 32, 32, 1, 10, type, path);
    set_train_params(sess, 100, 16, 16, 0.01);
    init_normal(sess, 0, 0.1);
    init_session(sess, "./data/fmnist/train.txt", "./data/fmnist/train_label.txt");
    train(sess, 0);
}

void lenet5_fmnist_detect(char*type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(6, 5, 1, 0, 1, 0, "logistic");
    Layer *l2 = make_maxpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(16, 5, 1, 0, 1, 0, "logistic");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_convolutional_layer(120, 5, 1, 0, 1, 0, "logistic");
    Layer *l6 = make_connect_layer(84, 1, "logistic");
    Layer *l7 = make_connect_layer(10, 1, "logistic");
    Layer *l8 = make_softmax_layer(10);
    Layer *l9 = make_mse_layer(10);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);
    append_layer2grpah(g, l9);
    Session *sess = create_session(g, 32, 32, 1, 10, type, path);
    set_detect_params(sess);
    init_normal(sess, 0, 0.1);
    init_session(sess, "./data/fmnist/test.txt", "./data/fmnist/test_label.txt");
    detect_classification(sess, 0);
}
