#include "alexnet.h"

void alexnet(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(96, 11, 4, 0, 1, 0, "relu");
    Layer *l2 = make_maxpool_layer(3, 2, 0);
    Layer *l3 = make_convolutional_layer(256, 5, 1, 2, 1, 0, "relu");
    Layer *l4 = make_maxpool_layer(3, 2, 0);
    Layer *l5 = make_convolutional_layer(384, 3, 1, 1, 1, 0, "relu");
    Layer *l6 = make_convolutional_layer(384, 3, 1, 1, 1, 0, "relu");
    Layer *l7 = make_convolutional_layer(256, 3, 1, 1, 1, 0, "relu");
    Layer *l8 = make_maxpool_layer(3, 2, 0);
    Layer *l9 = make_connect_layer(4096, 1, "relu");
    Layer *l10 = make_dropout_layer(0.5);
    Layer *l11 = make_connect_layer(4096, 1, "relu");
    Layer *l12 = make_dropout_layer(0.5);
    Layer *l13 = make_connect_layer(2, 1, "linear");
    Layer *l14 = make_softmax_layer(2);
    Layer *l15 = make_mse_layer(2);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);
    append_layer2grpah(g, l9);
    append_layer2grpah(g, l10);
    append_layer2grpah(g, l11);
    append_layer2grpah(g, l12);
    append_layer2grpah(g, l13);
    append_layer2grpah(g, l14);
    append_layer2grpah(g, l15);
    Session *sess = create_session(g, 224, 224, 1, 2, type, path);
    set_train_params(sess, 200, 32, 32, 0.0001);
    init_normal(sess, 0, 0.01);
    init_session(sess, "./data/xray/train/train.txt", "./data/xray/train/label.txt");
    train(sess, 0);
}

void alexnet_detect(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(96, 11, 4, 0, 1, 0, "relu");
    Layer *l2 = make_maxpool_layer(3, 2, 0);
    Layer *l3 = make_convolutional_layer(256, 5, 1, 2, 1, 0, "relu");
    Layer *l4 = make_maxpool_layer(3, 2, 0);
    Layer *l5 = make_convolutional_layer(384, 3, 1, 1, 1, 0, "relu");
    Layer *l6 = make_convolutional_layer(384, 3, 1, 1, 1, 0, "relu");
    Layer *l7 = make_convolutional_layer(256, 3, 1, 1, 1, 0, "relu");
    Layer *l8 = make_maxpool_layer(3, 2, 0);
    Layer *l9 = make_connect_layer(4096, 1, "relu");
    Layer *l10 = make_dropout_layer(0.5);
    Layer *l11 = make_connect_layer(4096, 1, "relu");
    Layer *l12 = make_dropout_layer(0.5);
    Layer *l13 = make_connect_layer(2, 1, "linear");
    Layer *l14 = make_softmax_layer(2);
    Layer *l15 = make_mse_layer(2);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);
    append_layer2grpah(g, l9);
    append_layer2grpah(g, l10);
    append_layer2grpah(g, l11);
    append_layer2grpah(g, l12);
    append_layer2grpah(g, l13);
    append_layer2grpah(g, l14);
    append_layer2grpah(g, l15);
    Session *sess = create_session(g, 224, 224, 3, 2, type, path);
    set_detect_params(sess);
    init_normal(sess, 0, 0.01);
    init_session(sess, "./data/xray/test/test.txt", "./data/xray/test/label.txt");
    detect_classification(sess, 0);
}

