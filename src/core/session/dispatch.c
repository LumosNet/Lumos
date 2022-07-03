#include "dispatch.h"

void session_run(Session *sess, float learning_rate)
{
    fprintf(stderr, "\nSession Start To Running\n");
    sess->learning_rate = learning_rate;
    for (int i = 0; i < sess->epoch; ++i){
        int sub_epochs = (int)(sess->train_data_num / sess->batch);
        int sub_batchs = (int)(sess->batch / sess->subdivision);
        for (int j = 0; j < sub_epochs; ++j){
            for (int k = 0; k < sub_batchs; ++k){
                load_data(sess, j*sess->batch+k*sess->subdivision, sess->subdivision);
                load_label(sess, j*sess->batch+k*sess->subdivision, sess->subdivision);
                forward_session(sess);
                backward_session(sess);
                update_session(sess);
            }
            memcpy(sess->weights, sess->update_weights, sess->weights_size*sizeof(float));
        }
    }
    save_weigths(sess, "./backup/Lumos.w");
    fprintf(stderr, "\nWeights Saved To: ./backup/Lumos.w\n");
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
        fill_cpu(sess->workspace, sess->workspace_size, 0, 1);
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
        fill_cpu(sess->workspace, sess->workspace_size, 0, 1);
        l->backward(*l, sess->subdivision, delta);
        delta = l->delta;
    }
}

void update_session(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    Layer *l;
    float rate = -sess->learning_rate / (float)sess->batch;
    float *delta = NULL;
    for (int i = graph->layer_num-1; i >= 0; --i){
        l = layers[i];
        if (l->update){
            fill_cpu(sess->workspace, sess->workspace_size, 0, 1);
            l->update(*l, rate, sess->subdivision, delta);
        }
        delta = l->delta;
    }
}

void create_run_scene(Session *sess, int h, int w, int c, int label_num, char *dataset_list_file, char *label_list_file)
{
    set_input_dimension(sess, h, w, c);
    bind_train_data(sess, dataset_list_file);
    bind_label(sess, label_num, label_list_file);
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
    get_workspace_size(sess);
    statistics_memory_occupy_size(sess);
    create_run_memory(sess);
    set_graph_memory(sess);
    create_weights_memory(sess);
    set_graph_weight(sess);
    init_weights(sess, weights_file);
    create_label_memory(sess);
    set_label(sess);
    set_maxpool_index_memory(sess);
}
