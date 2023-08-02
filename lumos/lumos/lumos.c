#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avgpool_layer.h"
#include "connect_layer.h"
#include "convolutional_layer.h"
#include "dropout_layer.h"
#include "im2col.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "shortcut.h"
#include "softmax.h"
#include "mse_layer.h"
#include "graph.h"
#include "layer.h"
#include "weights_init.h"
#include "dispatch.h"
#include "manager.h"
#include "session.h"

#define VERSION "0.1"

void lumos(int argc, char **argv)
{
    if (argc <= 1){
        fprintf(stderr, "Lumos fatal: No input option; use option --help for more information");
        return;
    }
    if (0 == strcmp(argv[1], "--version") || 0 == strcmp(argv[1], "-v"))
    {
        char version[] = VERSION;
        fprintf(stderr, "Lumos version: v%s\n", version);
        fprintf(stderr, "This is free software; see the source for copying conditions.  There is NO\n");
        fprintf(stderr, "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n");
    }
    else if (0 == strcmp(argv[1], "--train") || 0 == strcmp(argv[1], "-t"))
    {
        if (argc <= 6){
            fprintf(stderr, "Lumos fatal: Wrong traing option; use option --help for more information\n");
            return;
        }
        char *coretype = argv[2];
        char *demofile = argv[3];
        char *datapath = argv[4];
        char *labelpath = argv[5];
        char *weightfile = argv[6];
        Session *sess = load_session_json(demofile, coretype);
        bind_train_data(sess, datapath);
        bind_train_label(sess, labelpath);
        if (0 == strcmp(weightfile, "null")){
            init_train_scene(sess, NULL);
        } else {
            init_train_scene(sess, weightfile);
        }
        session_train(sess, "./build/lumos.w");
    }
    else if (0 == strcmp(argv[1], "--detect") || 0 == strcmp(argv[1], "-d"))
    {
        if (argc <= 6){
            fprintf(stderr, "Lumos fatal: Wrong traing option; use option --help for more information\n");
            return;
        }
        char *coretype = argv[2];
        char *demofile = argv[3];
        char *datapath = argv[4];
        char *labelpath = argv[5];
        char *weightfile = argv[6];
        if (0 == strcmp(weightfile, "null")){
            fprintf(stderr, "Lumos fatal: NULL weights path; Please input correct weights file path\n");
            return;
        }
        Session *sess = load_session_json(demofile, coretype);
        bind_test_data(sess, datapath);
        bind_test_label(sess, labelpath);
        init_test_scene(sess, weightfile);
        session_test(sess);
    }
    else if (0 == strcmp(argv[1], "--help") || 0 == strcmp(argv[1], "-h"))
    {
        fprintf(stderr, "Usage commands:\n");
        fprintf(stderr, "    --version or -v : To get version\n");
        fprintf(stderr, "    --path or -p : View The installation path\n");
        fprintf(stderr, "Thank you for using Lumos deeplearning framework.\n");
    }
    else
    {
        fprintf(stderr, "Lumos: \e[0;31merror\e[0m: unrecognized command line option '%s'\n", argv[1]);
        fprintf(stderr, "compilation terminated.\n");
        fprintf(stderr, "use --help or -h to get available commands.\n");
    }
}

int main(int argc, char **argv)
{
    // lumos(argc, argv);

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

    Initializer init = {0};
    init = he_initializer();
    Session *sess = create_session("cpu", init);
    bind_graph(sess, graph);
    sess->height = 1;
    sess->width = 2;
    sess->channel = 1;
    sess->epoch = 100;
    sess->batch = 1;
    sess->subdivision = 1;
    sess->learning_rate = 0.1;
    sess->label_num = 2;

    float *weights = calloc(16, sizeof(float));
    weights[0] = 0.1;
    weights[1] = 0.2;
    weights[2] = 0.4;
    weights[3] = -0.1;
    weights[4] = 0.2;
    weights[5] = 0.1;
    weights[6] = -0.3;
    weights[7] = 0.5;
    weights[8] = 0.1;
    weights[9] = 0.2;
    weights[10] = 0.4;
    weights[11] = -0.1;
    weights[12] = 0.2;
    weights[13] = 0.1;
    weights[14] = -0.3;
    weights[15] = 0.5;

    sess->weights = weights;
    sess->weights_size = 16;
    save_weigths(sess, "xor.w");
    return 0;
}
