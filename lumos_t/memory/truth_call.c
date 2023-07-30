#include "truth_call.h"

void call_truth(void **params, void **ret)
{
    char *graphF = params[0];
    float *truth = params[1];
    Session *sess = load_session_json(graphF, "cpu");
    init_train_scene(sess, NULL);
    Graph *graph = sess->graph;
    Layer *l = graph->layers[graph->layer_num - 1];
    for (int i = 0; i < sess->truth_num; ++i){
        l->truth[i] = truth[i];
    }
    ret[0] = sess->truth;
}
