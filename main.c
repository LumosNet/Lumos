#include <stdio.h>
#include <stdlib.h>

#include "cpu.h"
#include "graph.h"
#include "layer.h"
#include "im2col_layer.h"
#include "connect_layer.h"
#include "mse_layer.h"
#include "session.h"
#include "manager.h"
#include "dispatch.h"

#include "bias_gpu.h"
#include "pooling.h"

void mnist_label2truth(char **label, float *truth)
{
    int x = atoi(label[0]);
    one_hot_encoding(10, x, truth);
}

void mnist_process_test_information(char **label, float *truth, float *predict, float loss, char *data_path)
{
    fprintf(stderr, "Test Data Path: %s\n", data_path);
    fprintf(stderr, "Label:   %s\n", label[0]);
    fprintf(stderr, "Truth:       Predict:\n");
    for (int i = 0; i < 10; ++i){
        printf("%f     %f\n", truth[i], predict[i]);
    }
    fprintf(stderr, "Loss:    %f\n\n", loss);
}

void full_connect_mnist () {
    Graph *graph = create_graph("Lumos", 5);
    Layer *l1 = make_im2col_layer(1);
    Layer *l2 = make_connect_layer(128, 1, "logistic", "guass");
    Layer *l3 = make_connect_layer(64, 1, "logistic", "guass");
    Layer *l4 = make_connect_layer(10, 1, "logistic", "guass");
    Layer *l5 = make_mse_layer(10);
    append_layer2grpah(graph, l1);
    append_layer2grpah(graph, l2);
    append_layer2grpah(graph, l3);
    append_layer2grpah(graph, l4);
    append_layer2grpah(graph, l5);

    Session *sess = create_session();
    bind_graph(sess, graph);
    // create_train_scene(sess, 28, 28, 1, 1, 10, mnist_label2truth, "./data/train.txt", "./data/train_label.txt");
    create_train_scene(sess, 28, 28, 1, 1, 10, mnist_label2truth, "/usr/local/lumos/data/mnist/train.txt", "/usr/local/lumos/data/mnist/train_label.txt");
    init_train_scene(sess, 2000, 4, 4, NULL);
    session_train(sess, 0.01, "./lumos.w");

    // Session *t_sess = create_session();
    // bind_graph(t_sess, graph);
    // create_test_scene(t_sess, 28, 28, 1, 1, 10, mnist_label2truth, "/usr/local/lumos/data/mnist/train.txt", "/usr/local/lumos/data/mnist/train_label.txt");
    // init_test_scene(t_sess, "./lumos.w");
    // session_test(t_sess, mnist_process_test_information);
}

int main()
{
    // full_connect_mnist();
    int ksize = 2;
    int pad = 1;
    int stride = 1;
    int h[1], w[1], c[1];
    float *img = load_image_data("./data/1.jpg", w, h, c);
    int out_h = (h[0] + 2 * pad - ksize) / stride + 1;
    int out_w = (w[0] + 2 * pad - ksize) / stride + 1;
    float *space = malloc(out_h*out_w*c[0]*sizeof(float));
    avgpool(img, h[0], w[0], c[0], ksize, stride, pad, space);
    save_image_data(space, out_w, out_h, c[0], "./data/t.png");
    return 0;
}
