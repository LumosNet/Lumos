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
    
}

void create_delta_memory(Session sess);