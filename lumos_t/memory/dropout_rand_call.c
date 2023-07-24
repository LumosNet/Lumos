#include "dropout_rand_call.h"

void call_dropout_rand(void **params, void **ret)
{
    char *graphF = params[0];
    float *dropout_rand = params[1];
    Session *sess = load_session_json(graphF, "cpu");
    init_train_scene(sess, NULL);
    Graph *graph = sess->graph;
    Layer *l = NULL;
    int offset = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        l = graph->layers[i];
        if (l->type == DROPOUT){
            float *rand = dropout_rand + offset;
            for (int j = 0; j < l->inputs*sess->subdivision; ++j){
                l->dropout_rand[j] = rand[j];
            }
            offset += l->inputs*sess->subdivision;
        }
    }
    ret[0] = sess->dropout_rand;
}
