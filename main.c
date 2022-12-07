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

#include "binary_benchmark.h"

void lenet_label2truth(char **label, float *truth)
{
    int x = atoi(label[0]);
    one_hot_encoding(10, x, truth);
}

void lenet_process_test_information(char **label, float *truth, float *predict, float loss, char *data_path)
{
    fprintf(stderr, "Test Data Path: %s\n", data_path);
    fprintf(stderr, "Label:   %s\n", label[0]);
    fprintf(stderr, "Truth:       Predict:\n");
    for (int i = 0; i < 10; ++i){
        printf("%f     %f\n", truth[i], predict[i]);
    }
    fprintf(stderr, "Loss:    %f\n\n", loss);
}

void lenet() {
    Graph *graph = create_graph("Lumos", 9);
    Layer *l1 = make_convolutional_layer(6, 5, 1, 0, 1, 1, "logistic", "guass");
    Layer *l2 = make_avgpool_layer(2);
    Layer *l3 = make_convolutional_layer(16, 5, 1, 0, 1, 1, "logistic", "guass");
    Layer *l4 = make_avgpool_layer(2);
    Layer *l5 = make_convolutional_layer(120, 5, 1, 0, 1, 1, "logistic", "guass");
    Layer *l6 = make_im2col_layer(1);
    Layer *l7 = make_connect_layer(84, 1, "logistic", "guass");
    Layer *l8 = make_connect_layer(10, 1, "logistic", "guass");
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
    init_train_scene(sess, 500, 16, 16, NULL);
    // session_train(sess, 0.1, "./lumos.w");

    // Session *t_sess = create_session();
    // bind_graph(t_sess, graph);
    // create_test_scene(t_sess, 32, 32, 1, 1, 10, lenet_label2truth, "/usr/local/lumos/data/mnist/test.txt", "/usr/local/lumos/data/mnist/test_label.txt");
    // init_test_scene(t_sess, "./lumos.w");
    // session_test(t_sess, lenet_process_test_information);
}


int main()
{
    // lenet();
    // int ksize = 2;
    // int pad = 0;
    // int stride = ksize;
    // int h[1], w[1], c[1];
    // float *img = load_image_data("./data/1.jpg", w, h, c);
    // int out_h = (h[0] + 2 * pad - ksize) / stride + 1;
    // int out_w = (w[0] + 2 * pad - ksize) / stride + 1;
    // float *space = malloc(out_h*out_w*c[0]*sizeof(float));
    // int *index = malloc(out_h*out_w*c[0]*sizeof(int));
    // // avgpool(img, h[0], w[0], c[0], ksize, stride, pad, space);
    // maxpool(img, h[0], w[0], c[0], ksize, stride, pad, space, index);
    // save_image_data(space, out_w, out_h, c[0], "./data/t.png");
    // int head[1] = {12};
    // FILE *fp = fopen("test.bin", "wb");
    // fwrite(head, sizeof(int), 1, fp);
    // fclose(fp);
    void *buffer;
    buffer = get_binary_benchmark("file_name.bin");
    return 0;
}
