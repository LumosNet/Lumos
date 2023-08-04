#include "loss_gpu_call.h"

void call_loss_gpu(void **params, void **ret)
{
    char *graphF = params[0];
    float *loss = params[1];
    Session *sess = load_session_json(graphF, "cpu");
    init_train_scene(sess, NULL);
    Graph *graph = sess->graph;
    Layer *l = graph->layers[graph->layer_num - 1];
    l->loss[0] = loss[0];
    ret[0] = sess->loss_gpu;
}
