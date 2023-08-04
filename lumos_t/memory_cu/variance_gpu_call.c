#include "variance_gpu_call.h"

void call_variance_gpu(void **params, void **ret)
{
    char *graphF = params[0];
    float *variance = params[1];
    Session *sess = load_session_json(graphF, "gpu");
    init_train_scene(sess, NULL);
    Graph *graph = sess->graph;
    Layer *l = NULL;
    int offset = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        l = graph->layers[i];
        if (l->batchnorm){
            float *variance_c = variance + offset;
            for (int j = 0; j < l->output_c * sess->subdivision; ++j){
                l->variance[j] = variance_c[j];
            }
            offset += l->output_c * sess->subdivision;
        }
    }
    ret[0] = sess->variance_gpu;
}
