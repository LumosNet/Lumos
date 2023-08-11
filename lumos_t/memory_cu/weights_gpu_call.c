#include "weights_gpu_call.h"

void call_weights_gpu(void **params, void **ret)
{
    char *graphF = params[0];
    float *weights = params[1];
    Session *sess = load_session_json(graphF, "gpu");
    init_train_scene(sess, NULL);
    Graph *graph = sess->graph;
    Layer *l = NULL;
    int offset = 0;
    float *weights_c;
    for (int i = 0; i < graph->layer_num; ++i){
        l = graph->layers[i];
        if (l->weights){
            weights_c = weights + offset;
            cudaMemcpy(l->kernel_weights_gpu, weights_c, l->kernel_weights_size*sizeof(float), cudaMemcpyDeviceToDevice);
            offset += l->kernel_weights_size;
            if (l->bias){
                weights_c = weights + offset;
                cudaMemcpy(l->bias_weights_gpu, weights_c, l->bias_weights_size*sizeof(float), cudaMemcpyDeviceToDevice);
                offset += l->bias_weights_size;
            }
            if (l->batchnorm){
                weights_c = weights + offset;
                cudaMemcpy(l->normalize_weights_gpu, weights_c, l->normalize_weights_size*sizeof(float), cudaMemcpyDeviceToDevice);
                offset += l->normalize_weights_size;
            }
        }
    }
    ret[0] = sess->weights_gpu;
}
