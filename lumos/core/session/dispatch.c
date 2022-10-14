#include "dispatch.h"

void session_train(Session *sess, float learning_rate, char *weights_path)
{
    fprintf(stderr, "\nSession Start To Running\n");
    clock_t start, final;
    double run_time = 0;
    sess->learning_rate = learning_rate;
    for (int i = 0; i < sess->epoch; ++i)
    {
        fprintf(stderr, "\n\nEpoch %d/%d\n", i + 1, sess->epoch);
        start = clock();
        int sub_epochs = (int)(sess->train_data_num / sess->batch);
        int sub_batchs = (int)(sess->batch / sess->subdivision);
        for (int j = 0; j < sub_epochs; ++j)
        {
            for (int k = 0; k < sub_batchs; ++k)
            {
                load_train_data(sess, j * sess->batch + k * sess->subdivision, sess->subdivision);
                load_train_label(sess, j * sess->batch + k * sess->subdivision, sess->subdivision);
                forward_session(sess);
                backward_session(sess);
                final = clock();
                run_time = (double)(final - start) / CLOCKS_PER_SEC;
                progress_bar(j * sub_batchs + k + 1, sub_epochs * sub_batchs, run_time, sess->loss[0]);
            }
            memcpy(sess->weights, sess->update_weights, sess->weights_size * sizeof(float));
        }
    }
    fprintf(stderr, "\n\nSession Training Finished\n");
    save_weigths(sess, weights_path);
    fprintf(stderr, "\nWeights Saved To: %s\n\n", weights_path);
}

void session_test(Session *sess, ProcessTestInformation process_test_information)
{
    fprintf(stderr, "\nSession Start To Detect Test Cases\n");
    for (int i = 0; i < sess->test_data_num; ++i)
    {
        load_test_data(sess, i);
        char **label = load_test_label(sess, i);
        forward_session(sess);
        Graph *graph = sess->graph;
        Layer **layers = graph->layers;
        sess->predicts = layers[graph->layer_num - 2]->output;
        process_test_information(label, sess->truth, sess->predicts, sess->loss[0], sess->test_data_paths[i]);
    }
    fprintf(stderr, "\nSession Testing Finished\n\n");
}

void forward_session(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    Layer *l;
    float *input = sess->input;
    for (int i = 0; i < graph->layer_num; ++i)
    {
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
    float rate = -sess->learning_rate / (float)sess->batch;
    float *delta = NULL;
    for (int i = graph->layer_num - 1; i >= 0; --i)
    {
        l = layers[i];
        l->backward(*l, rate, sess->subdivision, delta);
        delta = l->delta;
    }
}

void create_train_scene(Session *sess, int h, int w, int c, int label_num, int truth_num, Label2Truth func, char *dataset_list_file, char *label_list_file)
{
    set_input_dimension(sess, h, w, c);
    bind_train_data(sess, dataset_list_file);
    bind_train_label(sess, label_num, label_list_file);
    bind_label2truth_func(sess, truth_num, func);
}

void create_test_scene(Session *sess, int h, int w, int c, int label_num, int truth_num, Label2Truth func, char *dataset_list_file, char *label_list_file)
{
    set_input_dimension(sess, h, w, c);
    bind_test_data(sess, dataset_list_file);
    bind_test_label(sess, label_num, label_list_file);
    bind_label2truth_func(sess, truth_num, func);
}

void init_train_scene(Session *sess, int epoch, int batch, int subdivision, char *weights_file)
{
    fprintf(stderr, "\nEpoch   Batch   Subdivision\n");
    fprintf(stderr, "%3d     %3d     %3d\n", epoch, batch, subdivision);
    sess->epoch = epoch;
    sess->batch = batch;
    sess->subdivision = subdivision;
    sess->input = calloc(sess->subdivision * sess->height * sess->width * sess->channel, sizeof(float));
    init_graph(sess->graph, sess->width, sess->height, sess->channel);
    get_workspace_size(sess);
    statistics_memory_occupy_size(sess);
    create_run_memory(sess);
    set_graph_memory(sess);
    create_weights_memory(sess);
    set_graph_weight(sess);
    init_weights(sess, weights_file);
    create_label_memory(sess);
    create_loss_memory(sess);
    set_loss_memory(sess);
    set_label(sess);
    create_truth_memory(sess);
    set_truth_memory(sess);
    set_maxpool_index_memory(sess);
}

void init_test_scene(Session *sess, char *weights_file)
{
    if (weights_file == NULL)
        return;
    sess->epoch = 1;
    sess->batch = 1;
    sess->subdivision = 1;
    sess->input = calloc(sess->subdivision * sess->height * sess->width * sess->channel, sizeof(float));
    init_graph(sess->graph, sess->width, sess->height, sess->channel);
    get_workspace_size(sess);
    statistics_memory_occupy_size(sess);
    create_run_memory(sess);
    set_graph_memory(sess);
    create_weights_memory(sess);
    set_graph_weight(sess);
    init_weights(sess, weights_file);
    create_label_memory(sess);
    create_loss_memory(sess);
    set_loss_memory(sess);
    set_label(sess);
    create_truth_memory(sess);
    set_truth_memory(sess);
    set_maxpool_index_memory(sess);
}
