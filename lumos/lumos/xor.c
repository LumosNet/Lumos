#include "xor.h"

void xor_label2truth(char **label, float *truth)
{
    int x = atoi(label[0]);
    one_hot_encoding(1, x, truth);
}

void xor_process_test_information(char **label, float *truth, float *predict, float loss, char *data_path)
{
    fprintf(stderr, "Test Data Path: %s\n", data_path);
    fprintf(stderr, "Label:   %s\n", label[0]);
    fprintf(stderr, "Truth:   %f\n", truth[0]);
    fprintf(stderr, "Predict: %f\n", predict[0]);
    fprintf(stderr, "Loss:    %f\n\n", loss);
}

void xor()
{
    Graph *graph = create_graph("Lumos", 5);
    Layer *l1 = make_im2col_layer(1);
    Layer *l2 = make_connect_layer(4, 1, "logistic");
    Layer *l3 = make_connect_layer(2, 1, "logistic");
    Layer *l4 = make_connect_layer(1, 1, "logistic");
    Layer *l5 = make_mse_layer(1);
    append_layer2grpah(graph, l1);
    append_layer2grpah(graph, l2);
    append_layer2grpah(graph, l3);
    append_layer2grpah(graph, l4);
    append_layer2grpah(graph, l5);

    Session *sess = create_session();
    bind_graph(sess, graph);
    create_train_scene(sess, 1, 2, 1, 1, 1, xor_label2truth, "./demos/xor/data.txt", "./demos/xor/label.txt");
    init_train_scene(sess, 4500, 4, 2, NULL);
    session_train(sess, 0.1);

    Session *t_sess = create_session();
    bind_graph(t_sess, graph);
    create_test_scene(t_sess, 1, 2, 1, 1, 1, xor_label2truth, "./demos/xor/test.txt", "./demos/xor/label.txt");
    init_test_scene(t_sess, "./Lumos.w");
    session_test(t_sess, xor_process_test_information);
}
