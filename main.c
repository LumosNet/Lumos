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

int main(int argc, char **argv)
{
    Graph *graph = create_graph("Lumos", 9);
    Layer *l1 = make_convolutional_layer(16, 3, 1, 1, 1, 1, "relu");
    Layer *l2 = make_avgpool_layer(2);
    Layer *l3 = make_convolutional_layer(16, 3, 1, 1, 1, 1, "relu");
    Layer *l4 = make_maxpool_layer(2);
    Layer *l5 = make_im2col_layer(1);
    Layer *l6 = make_connect_layer(128, 1, "relu");
    Layer *l7 = make_connect_layer(64, 1, "relu");
    Layer *l8 = make_connect_layer(32, 1, "relu");
    Layer *l9 = make_connect_layer(1, 1, "relu");
    Layer *l10 = make_mse_layer(2);
    append_layer2grpah(graph, l1);
    append_layer2grpah(graph, l2);
    append_layer2grpah(graph, l3);
    append_layer2grpah(graph, l4);
    append_layer2grpah(graph, l5);
    append_layer2grpah(graph, l6);
    append_layer2grpah(graph, l7);
    append_layer2grpah(graph, l8);
    append_layer2grpah(graph, l9);
    append_layer2grpah(graph, l10);

    Session *sess = create_session();
    bind_graph(sess, graph);
    create_run_scene(sess, 28, 28, 1, 1, "./demo/xor/data.txt", "./demo/xor/label.txt");
    init_run_scene(sess, 20, 25, 5, NULL);
    session_run(sess);
}

// def initialize_parameters_random(layers_dims):
//     """
//     Arguments:
//     layer_dims -- python array (list) containing the size of each layer.
//     Returns:
//     parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
//                     W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
//                     b1 -- bias vector of shape (layers_dims[1], 1)
//                     ...
//                     WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
//                     bL -- bias vector of shape (layers_dims[L], 1)
//     """
//     np.random.seed(3)  # This seed makes sure your "random" numbers will be the as ours
//     parameters = {}
//     L = len(layers_dims)  # integer representing the number of layers
//     for l in range(1, L):
//         parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])*0.01
//         parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
//     return parameters
