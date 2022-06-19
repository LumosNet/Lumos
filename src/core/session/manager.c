#include "manager.h"

void create_run_memory(Session *sess)
{
    create_workspace_memory(sess);
    create_output_memory(sess);
    create_delta_memory(sess);
}

void create_workspace_memory(Session *sess)
{
    sess->workspace = calloc(sess->workspace_size, sizeof(float));
    fprintf(stderr, "Apply For The Commond Work Space\n");
}

void create_input_memory(Session *sess)
{
    int inputs = sess->subdivision*sess->height*sess->width*sess->channel;
    sess->input = calloc(inputs, sizeof(float));
}

void create_output_memory(Session *sess)
{
    Graph *graph = sess->graph;
    int outputs = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = graph->layers[i];
        outputs += l->outputs;
    }
    sess->output = calloc(outputs*sess->subdivision, sizeof(float));
    fprintf(stderr, "Apply For Layers Output Data Space\n");
}

void create_delta_memory(Session *sess)
{
    Graph *graph = sess->graph;
    int deltas = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = graph->layers[i];
        deltas += l->deltas;
    }
    sess->layer_delta = calloc(deltas*sess->subdivision, sizeof(float));
    fprintf(stderr, "APPly For Layers Delta Data Space\n");
}

void set_graph_memory(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    int output_offset = 0;
    int delta_offset = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = layers[i];
        l->output = sess->output+output_offset;
        l->delta = sess->layer_delta+delta_offset;
        l->workspace = sess->workspace;
        output_offset += l->outputs*sess->subdivision;
        delta_offset += l->deltas*sess->subdivision;
    }
    fprintf(stderr, "\nDistribut Running Memory To Each Layer\n");
}

void set_graph_weight(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    int weights_offset = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = layers[i];
        if (l->weights){
            l->kernel_weights = sess->weights+weights_offset;
            l->update_kernel_weights = sess->update_weights+weights_offset;
            weights_offset += l->kernel_weights_size;
            if (l->bias){
                l->bias_weights = sess->weights+weights_offset;
                l->update_bias_weights = sess->update_weights+weights_offset;
                weights_offset += l->bias_weights_size;
            }
        }
    }
    fprintf(stderr, "\nDistribut Weights To Each Layer\n");
}
