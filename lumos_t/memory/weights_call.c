#include "weights_call.h"

void call_weights(void **params, void **ret)
{
    char *graphF = params[0];
    float *weights = params[1];
    Session *sess = load_session_json(graphF, "cpu");
    init_train_scene(sess, NULL);
    Graph *graph = sess->graph;
    Layer *l = NULL;
    int offset = 0;
    float *weights_c;
    for (int i = 0; i < graph->layer_num; ++i){
        l = graph->layers[i];
        if (l->weights){
            
        }
    }
}
