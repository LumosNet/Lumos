#include <stdio.h>
#include <stdlib.h>

#include "session.h"
#include "manager.h"
#include "dispatch.h"
#include "graph.h"
#include "layer.h"
#include "convolutional_layer.h"
#include "connect_layer.h"
#include "im2col_layer.h"
#include "maxpool_layer.h"
#include "avgpool_layer.h"
#include "mse_layer.h"
#include "test.h"

int main(int argc, char **argv)
{
    Graph *graph = create_graph("Lumos", 4);
    Layer *l1 = make_im2col_layer(1);
    Layer *l2 = make_connect_layer(4, 1, "logistic");
    Layer *l3 = make_connect_layer(1, 1, "logistic");
    Layer *l4 = make_mse_layer(1);
    append_layer2grpah(graph, l1);
    append_layer2grpah(graph, l2);
    append_layer2grpah(graph, l3);
    append_layer2grpah(graph, l4);

    Session *sess = create_session();
    bind_graph(sess, graph);
    create_run_scene(sess, 1, 2, 1, 1, "./demo/xor/data.txt", "./demo/xor/label.txt");
    init_run_scene(sess, 1000000, 4, 4, NULL);
    session_run(sess, 0.1);

    return 0;
}
