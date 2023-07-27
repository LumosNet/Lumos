#include "maxpool_index_call.h"

void call_maxpool_index(void **params, void **ret)
{
    char *graphF = params[0];
    float *index = params[1];
    Session *sess = load_session_json(graphF, "cpu");
    init_train_scene(sess, NULL);
    Graph *graph = sess->graph;
    Layer *l = NULL;
    int offset = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        l = graph->layers[i];
        if (l->type == MAXPOOL){
            float *max_index = index + offset;
            for (int j = 0; j < l->outputs*sess->subdivision; ++j){
                l->maxpool_index[j] = max_index[j];
            }
            offset += l->outputs*sess->subdivision;
        }
    }
    ret[0] = sess->maxpool_index;
}
