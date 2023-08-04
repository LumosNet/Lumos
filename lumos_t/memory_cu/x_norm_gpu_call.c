#include "x_norm_gpu_call.h"

void call_x_norm_gpu(void **params, void **ret)
{
    char *graphF = params[0];
    float *x_norm = params[1];
    Session *sess = load_session_json(graphF, "gpu");
    init_train_scene(sess, NULL);
    Graph *graph = sess->graph;
    Layer *l = NULL;
    int offset = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        l = graph->layers[i];
        if (l->batchnorm){
            float *x_norm_c = x_norm + offset;
            for (int j = 0; j < l->outputs * sess->subdivision; ++j){
                l->x_norm[j] = x_norm_c[j];
            }
            offset += l->outputs * sess->subdivision;
        }
    }
    ret[0] = sess->x_norm_gpu;
}
