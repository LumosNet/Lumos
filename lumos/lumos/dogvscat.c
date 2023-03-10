#include "dogvscat.h"

void dogvscat_label2truth(char **label, float *truth)
{
    int x = atoi(label[0]);
    one_hot_encoding(2, x, truth);
}

void dogvscat_process_test_information(char **label, float *truth, float *predict, float loss, char *data_path)
{
    fprintf(stderr, "Test Data Path: %s\n", data_path);
    fprintf(stderr, "Label:   %s\n", label[0]);
    fprintf(stderr, "Truth:       Predict:\n");
    for (int i = 0; i < 2; ++i){
        printf("%f     %f\n", truth[i], predict[i]);
    }
    fprintf(stderr, "Loss:    %f\n\n", loss);
}

void dogvscat()
{
    Graph *graph = create_graph("Lumos", 9);
    Layer *l1 = make_convolutional_layer(16, 3, 3, 1, 1, 0, "logistic", "guass");
    Layer *l2 = make_maxpool_layer(2);
    Layer *l3 = make_convolutional_layer(16, 3, 3, 1, 1, 0, "logistic", "guass");
    Layer *l4 = make_maxpool_layer(2);
    Layer *l5 = make_im2col_layer(1);
    Layer *l6 = make_connect_layer(128, 1, "logistic", "guass");
    Layer *l7 = make_connect_layer(64, 1, "logistic", "guass");
    Layer *l8 = make_connect_layer(2, 1, "logistic", "guass");
    Layer *l9 = make_mse_layer(2);
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
    create_train_scene(sess, 200, 200, 3, 1, 2, dogvscat_label2truth, "./data/dogvscat/train.txt", "./data/dogvscat/train_label.txt");
    init_train_scene(sess, 5000, 16, 16, NULL);
    session_train(sess, 0.01, "./lumos.w");
}
