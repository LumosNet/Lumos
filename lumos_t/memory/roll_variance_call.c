#include "roll_variance_call.h"

void call_roll_variance(void **params, void **ret)
{
    char *graphF = params[0];
    float *roll_variance = params[1];
    Session *sess = load_session_json(graphF, "cpu");
    init_train_scene(sess, NULL);
    Graph *graph = sess->graph;
    Layer *l = NULL;
    int offset = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        l = graph->layers[i];
        if (l->batchnorm){
            float *r_variance = roll_variance + offset;
            for (int j = 0; j < l->output_c * sess->subdivision; ++j){
                l->rolling_variance[j] = r_variance[j];
            }
        }
        offset += l->output_c * sess->subdivision;
    }
    ret[0] = sess->roll_variance;
}