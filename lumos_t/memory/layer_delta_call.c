#include "layer_delta_call.h"

void call_layer_delta(void **params, void **ret)
{
    char *graphF = params[0];
    float *layer_delta = params[1];
    Session *sess = load_session_json(graphF, "cpu");
    init_train_scene(sess, NULL);
    Graph *graph = sess->graph;
    Layer *l = NULL;
    int offset = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        l = graph->layers[i];
        float *delta = layer_delta + offset;
        for (int j = 0; j < l->deltas * sess->subdivision; ++j){
            l->delta[j] = delta[j];
        }
        offset += l->deltas * sess->subdivision;
    }
    ret[0] = sess->layer_delta;
}
