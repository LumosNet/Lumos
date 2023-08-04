#include "normalize_mean_gpu_call.h"

void call_normalize_mean_gpu(void **params, void **ret)
{
    char *graphF = params[0];
    float *mean = params[1];
    Session *sess = load_session_json(graphF, "gpu");
    init_train_scene(sess, NULL);
    Graph *graph = sess->graph;
    Layer *l = NULL;
    int offset = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        l = graph->layers[i];
        float *norm_mean = mean + offset;
        if (l->batchnorm){
            for (int j = 0; j < l->output_c*sess->subdivision; ++j){
                l->mean[j] = norm_mean[j];
            }
        }
        offset += l->output_c * sess->subdivision;
    }
    ret[0] = sess->mean_gpu;
}
