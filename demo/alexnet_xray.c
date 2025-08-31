#include "alexnet_xray.h"

void alexnet_xray(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(96, 11, 4, 0, 1, "relu");
    Layer *l2 = make_maxpool_layer(3, 2, 0);
    Layer *l3 = make_convolutional_layer(256, 5, 1, 2, 1, "relu");
    Layer *l4 = make_maxpool_layer(3, 2, 0);
    Layer *l5 = make_convolutional_layer(384, 3, 1, 1, 1, "relu");
    Layer *l6 = make_convolutional_layer(384, 3, 1, 1, 1, "relu");
    Layer *l7 = make_convolutional_layer(256, 3, 1, 1, 1, "relu");
    Layer *l8 = make_maxpool_layer(3, 2, 0);
    Layer *l9 = make_im2col_layer();
    Layer *l10 = make_connect_layer(4096, 1, "relu");
    Layer *l11 = make_dropout_layer(0.5);
    Layer *l12 = make_connect_layer(4096, 1, "relu");
    Layer *l13 = make_dropout_layer(0.5);
    Layer *l14 = make_connect_layer(2, 1, "linear");
    Layer *l15 = make_softmax_layer(2);
    Layer *l16 = make_mse_layer(2);
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
    append_layer2grpah(g, l16);
    // init_kaiming_normal(l1, 0, "fan_in", "relu");
    // init_kaiming_normal(l3, 0, "fan_in", "relu");
    // init_kaiming_normal(l5, 0, "fan_in", "relu");
    // init_kaiming_normal(l6, 0, "fan_in", "relu");
    // init_kaiming_normal(l7, 0, "fan_in", "relu");
    init_kaiming_uniform(l10, 0, "fan_in", "relu");
    init_kaiming_uniform(l12, 0, "fan_in", "relu");
    init_kaiming_uniform(l14, 0, "fan_in", "relu");

    init_normal(l1, 0, 0.01);
    init_normal(l3, 0, 0.01);
    init_normal(l5, 0, 0.01);
    init_normal(l6, 0, 0.01);
    init_normal(l7, 0, 0.01);
    Session *sess = create_session(g, 224, 224, 1, 2, type, path);
    set_train_params(sess, 200, 32, 32, 0.0001);
    init_session(sess, "./data/xray/train.txt", "./data/xray/label.txt");
    train(sess);
}

void alexnet_xray_detect(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(96, 11, 4, 0, 1, "relu");
    Layer *l2 = make_maxpool_layer(3, 2, 0);
    Layer *l3 = make_convolutional_layer(256, 5, 1, 2, 1, "relu");
    Layer *l4 = make_maxpool_layer(3, 2, 0);
    Layer *l5 = make_convolutional_layer(384, 3, 1, 1, 1, "relu");
    Layer *l6 = make_convolutional_layer(384, 3, 1, 1, 1, "relu");
    Layer *l7 = make_convolutional_layer(256, 3, 1, 1, 1, "relu");
    Layer *l8 = make_maxpool_layer(3, 2, 0);
    Layer *l9 = make_im2col_layer();
    Layer *l10 = make_connect_layer(4096, 1, "relu");
    Layer *l11 = make_dropout_layer(0.5);
    Layer *l12 = make_connect_layer(4096, 1, "relu");
    Layer *l13 = make_dropout_layer(0.5);
    Layer *l14 = make_connect_layer(2, 1, "linear");
    Layer *l15 = make_softmax_layer(2);
    Layer *l16 = make_mse_layer(2);
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
    append_layer2grpah(g, l16);
    Session *sess = create_session(g, 224, 224, 1, 2, type, path);
    set_detect_params(sess);
    init_session(sess, "./data/xray/train.txt", "./data/xray/label.txt");
    detect_classification(sess);
}
