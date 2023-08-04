#include "update_weights_gpu_call.h"

void call_update_weights_gpu(void **params, void **ret)
{
    char *graphF = params[0];
    float *update_weights = params[1];
    Session *sess = load_session_json(graphF, "gpu");
    init_train_scene(sess, NULL);
    Graph *graph = sess->graph;
    Layer *l = NULL;
    int offset = 0;
    float *weights;
    for (int i = 0; i < graph->layer_num; ++i){
        l = graph->layers[i];
        if (l->weights){
            weights = update_weights + offset;
            for (int j = 0; j < l->kernel_weights_size; ++j){
                l->update_kernel_weights_gpu[j] = weights[j];
            }
            offset += l->kernel_weights_size;
            if (l->bias){
                weights = update_weights + offset;
                for (int j = 0; j < l->bias_weights_size; ++j){
                    l->update_bias_weights_gpu[j] = weights[j];
                }
                offset += l->bias_weights_size;
            }
            if (l->batchnorm){
                weights = update_weights + offset;
                for (int j = 0; j < l->normalize_weights_size; ++j){
                    l->update_normalize_weights_gpu[j] = weights[j];
                }
                offset += l->normalize_weights_size;
            }
        }
    }
    ret[0] = sess->update_weights_gpu;
}
