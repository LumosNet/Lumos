#include "xor.h"

void xor(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_connect_layer(4, 1, "relu");
    Layer *l2 = make_connect_layer(2, 1, "relu");
    Layer *l3 = make_softmax_layer(2);
    Layer *l4 = make_mse_layer(2);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    init_uniform(l1, -1, 1);
    init_uniform(l2, -1, 1);
    Session *sess = create_session(g, 1, 2, 1, 2, type, path);
    set_train_params(sess, 50, 2, 2, 0.1);
    init_session(sess, "./data/xor/data.txt", "./data/xor/label.txt");
    train(sess, 0);
}

void xor_detect(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_connect_layer(4, 1, "relu");
    Layer *l2 = make_connect_layer(2, 1, "relu");
    Layer *l3 = make_softmax_layer(2);
    Layer *l4 = make_mse_layer(2);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    Session *sess = create_session(g, 1, 2, 1, 2, type, path);
    set_detect_params(sess);
    init_session(sess, "./data/xor/data.txt", "./data/xor/label.txt");
    detect_classification(sess, 0);
}
