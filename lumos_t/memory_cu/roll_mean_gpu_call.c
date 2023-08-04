#include "roll_mean_gpu_call.h"

void call_roll_mean_gpu(void **params, void **ret)
{
    char *graphF = params[0];
    float *roll_mean = params[1];
    Session *sess = load_session_json(graphF, "gpu");
    init_train_scene(sess, NULL);
    Graph *graph = sess->graph;
    Layer *l = NULL;
    int offset = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        l = graph->layers[i];
        if (l->batchnorm){
            float *r_mean = roll_mean + offset;
            for (int j = 0; j < l->output_c * sess->subdivision; ++j){
                l->rolling_mean[j] = r_mean[j];
            }
        }
        offset += l->output_c * sess->subdivision;
    }
    ret[0] = sess->roll_mean_gpu;
}
