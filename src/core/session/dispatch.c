#include "dispatch.h"

void session_run(Session *sess)
{
    fprintf(stderr, "\nSession Start To Running\n");
    clock_t start, final;
    double run_time = 0;
    for (int i = 0; i < sess->epoch; ++i){
        fprintf(stderr, "\nEpoch %d/%d\n", i+1, sess->epoch);
        int sub_epochs = (int)(sess->train_data_num / sess->batch);
        int sub_batchs = (int)(sess->batch / sess->subdivision);
        for (int j = 0; j < sub_epochs; ++j){
            for (int k = 0; k < sub_batchs; ++k){
                load_data(sess, j*sess->batch+k*sess->subdivision, sess->subdivision);
                start = clock();
                forward_session(sess);
                backward_session(sess);
                update_session(sess);
                final = clock();
                run_time = (double)(final-start) / CLOCKS_PER_SEC;
                fprintf(stderr, "%4d   RunTime\n", j*sub_batchs+k);
                fprintf(stderr, "        %.3lfs\n", run_time);
            }
            memcpy(sess->weights, sess->update_weights, sess->weights_size*sizeof(float));
        }
    }
    fprintf(stderr, "\n");
}

void forward_session(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    Layer *l;
    float *input = sess->input;
    for (int i = 0; i < graph->layer_num; ++i){
        l = layers[i];
        l->input = input;
        l->forward(*l, sess->subdivision);
        input = l->output;
    }
}

void backward_session(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    Layer *l;
    float *delta = NULL;
    for (int i = graph->layer_num-1; i >= 0; --i){
        l = layers[i];
        l->backward(*l, sess->subdivision, delta);
        delta = l->delta;
    }
}

void update_session(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    Layer *l;
    float rate = sess->learning_rate / sess->batch;
    float *delta = NULL;
    for (int i = graph->layer_num-1; i >= 0; --i){
        l = layers[i];
        if (l->update) l->update(*l, rate, delta);
        delta = l->delta;
    }
}

void create_run_scene(Session *sess, int h, int w, int c, char *dataset_list_file, char *label_list_file)
{
    set_input_dimension(sess, h, w, c);
    bind_train_data(sess, dataset_list_file);
    bind_label(sess, label_list_file);
}


void init_run_scene(Session *sess, int epoch, int batch, int subdivision, char *weights_file)
{
    fprintf(stderr, "\nEpoch   Batch   Subdivision\n");
    fprintf(stderr, "%3d     %3d     %3d\n", epoch, batch, subdivision);
    sess->epoch = epoch;
    sess->batch = batch;
    sess->subdivision = subdivision;
    sess->input = calloc(sess->subdivision*sess->height*sess->width*sess->channel, sizeof(float));
    init_graph(sess->graph, sess->width, sess->height, sess->channel);
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    int max_workspace_size = -1;
    int weights_size = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = layers[i];
        if (l->workspace_size > max_workspace_size){
            max_workspace_size = l->workspace_size;
        }
        weights_size += l->kernel_weights_size;
        weights_size += l->bias_weights_size;
    }
    sess->workspace_size = max_workspace_size;
    sess->weights_size = weights_size;
    statistics_memory_occupy_size(sess);
    create_run_memory(sess);
    set_graph_memory(sess);
    init_weights(sess, weights_file);
    set_graph_weight(sess);
    set_maxpool_index_memory(sess);
}
