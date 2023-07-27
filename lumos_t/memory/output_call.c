#include "output_call.h"

void call_output(void **params, void **ret)
{
    char *graphF = params[0];
    float *output = params[1];
    Session *sess = load_session_json(graphF, "cpu");
    init_train_scene(sess, NULL);
    Graph *graph = sess->graph;
    Layer *l = NULL;
    int offset = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        l = graph->layers[i];
        float *output_el = output + offset;
        memcpy(l->output, output_el, l->outputs*sess->subdivision*sizeof(float));
        offset += l->outputs * sess->subdivision;
    }
    ret[0] = sess->output;
}
