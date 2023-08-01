#include "xor_call.h"

void call_xor(void **params, void **ret)
{
    float *input = params[0];
    float *weights = params[1];
    Graph *graph = create_graph("xor", 5);
    Layer *l1 = make_im2col_layer(1);
    Layer *l2 = make_connect_layer(4, 1, 0, "relu");
    Layer *l3 = make_connect_layer(2, 1, 0, "relu");
    Layer *l4 = make_softmax_layer(2);
    Layer *l5 = make_mse_layer(2);
    append_layer2grpah(graph, l1);
    append_layer2grpah(graph, l2);
    append_layer2grpah(graph, l3);
    append_layer2grpah(graph, l4);
    append_layer2grpah(graph, l5);

    Session *sess = create_session();
    bind_graph(sess, graph);
    sess->height = 1;
    sess->width = 2;
    sess->channel = 1;
    sess->epoch = 100;
    sess->batch = 1;
    sess->subdivision = 1;
    sess->learning_rate = 0.1;
    sess->label_num = 2;
    bind_train_data(sess, "./data/xor/data.txt");
    bind_train_label(sess, "./data/xor/label.txt");
    init_train_scene(sess, "./lumos_t/conf/xor.w");
    session_train(sess, "./build/lumos.w");
    ret[0] = sess->output;
    ret[1] = sess->update_weights;
    ret[2] = sess->weights;
}