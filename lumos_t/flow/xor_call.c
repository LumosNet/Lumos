#include "xor_call.h"

void call_xor(void **params, void **ret)
{
    int *epoch = params[0];
    int *batch = params[1];
    int *subdivision = params[2];
    float *learning_rate = params[3];
    char *data_path = params[4];
    char *label_path = params[5];
    char *weights_path = params[6];
    Graph *graph = create_graph("xor", 5);
    Layer *l1 = make_im2col_layer();
    Layer *l2 = make_connect_layer(4, 1, 0, "relu");
    Layer *l3 = make_connect_layer(2, 1, 0, "relu");
    Layer *l4 = make_mse_layer(2);
    append_layer2grpah(graph, l1);
    append_layer2grpah(graph, l2);
    append_layer2grpah(graph, l3);
    append_layer2grpah(graph, l4);

    Initializer init = {0};
    init = he_initializer();
    Session *sess = create_session("cpu", init);
    bind_graph(sess, graph);
    sess->height = 1;
    sess->width = 2;
    sess->channel = 1;
    sess->epoch = epoch[0];
    sess->batch = batch[0];
    sess->subdivision = subdivision[0];
    sess->learning_rate = learning_rate[0];
    sess->label_num = 2;
    bind_train_data(sess, data_path);
    bind_train_label(sess, label_path);
    init_train_scene(sess, weights_path);
    session_train(sess, NULL);
    ret[0] = sess->output;
    ret[1] = sess->update_weights;
    ret[2] = sess->weights;
}