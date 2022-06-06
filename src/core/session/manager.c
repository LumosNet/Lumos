#include "manager.h"

void create_run_memory(Session sess)
{
    create_workspace_memory(sess);
    create_output_memory(sess);
    create_delta_memory(sess);
}

void create_workspace_memory(Session sess)
{
    Graph graph = sess.graph;
    Layer *layers = graph.layers;
    int max_workspace_size = -1;
    for (int i = 0; i < graph.layer_num; ++i){
        Layer l = layers[i];
        if (l.workspace_size > max_workspace_size){
            max_workspace_size = l.workspace_size;
        }
    }
    sess.workspace = calloc(max_workspace_size, sizeof(float));
}

void create_output_memory(Session sess)
{
    Graph graph = sess.graph;
    Layer *layers = graph.layers;
    int outputs = 0;
    for (int i = 0; i < graph.layer_num; ++i){
        Layer l = graph.layers[i];
        outputs += l.outputs;
    }
    sess.output = calloc(outputs, sizeof(float));
}

void create_delta_memory(Session sess)
{
    Graph graph = sess.graph;
    Layer *layers = graph.layers;
    int deltas = 0;
    for (int i = 0; i < graph.layer_num; ++i){
        Layer l = graph.layers[i];
        deltas += l.deltas;
    }
    sess.layer_delta = calloc(deltas, sizeof(float));
}

void set_graph_memory(Session sess)
{
    Graph graph = sess.graph;
    Layer *layers = graph.layers;
    int output_offset = 0;
    int delta_offset = 0;
    for (int i = 0; i < graph.layer_num; ++i){
        Layer l = layers[i];
        l.output = sess.output+output_offset;
        l.delta = sess.layer_delta+delta_offset;
        l.workspace = sess.workspace;
        output_offset += l.outputs;
        delta_offset += l.deltas;
    }
}

void set_graph_weight(Session sess)
{
    Graph graph = sess.graph;
    Layer *layers = graph.layers;
    int weights_offset = 0;
    for (int i = 0; i < graph.layer_num; ++i){
        Layer l = layers[i];
        if (l.weights){
            l.kernel_weights = sess.weights+weights_offset;
            weights_offset += l.kernel_weights_size;
            if (l.bias){
                l.bias_weights = sess.weights+weights_offset;
                weights_offset += l.bias_weights_size;
            }
        }
    }
}
