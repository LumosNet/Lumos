#include "dispatch.h"

void session_train(Session *sess, char *weights_path)
{
    fprintf(stderr, "\nSession Start To Running\n");
    clock_t start, final;
    double run_time = 0;
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
                if (sess->coretype == GPU){
                    
                    fill_gpu(sess->output_gpu, sess->output_size, 0, 1);
                    fill_gpu(sess->layer_delta_gpu, sess->delta_size, 0, 1);
                    fill_gpu(sess->workspace_gpu, sess->workspace_size, 0, 1);
                } else {
                    fill_cpu(sess->output, sess->output_size, 0, 1);
                    fill_cpu(sess->layer_delta, sess->delta_size, 0, 1);
                    fill_cpu(sess->workspace, sess->workspace_size, 0, 1);
                }
                load_train_data(sess, j * sess->batch + k * sess->subdivision, sess->subdivision);
                load_train_label(sess, j * sess->batch + k * sess->subdivision, sess->subdivision);
                forward_session(sess);
                backward_session(sess);
                final = clock();
                run_time = (double)(final - start) / CLOCKS_PER_SEC;
                if (sess->coretype == CPU) run_time /= 10;
                progress_bar(j * sub_batchs + k + 1, sub_epochs * sub_batchs, run_time, sess->loss[0]);
            }
            if (sess->coretype == GPU){
                cudaMemcpy(sess->weights_gpu, sess->update_weights_gpu, sess->weights_size*sizeof(float), cudaMemcpyDeviceToDevice);
            } else {
                memcpy(sess->weights, sess->update_weights, sess->weights_size*sizeof(float));
            }
        }
    }
    fprintf(stderr, "\n\nSession Training Finished\n");
    save_weigths(sess, weights_path);
    fprintf(stderr, "\nWeights Saved To: %s\n\n", weights_path);
}

void session_test(Session *sess)
{
    fprintf(stderr, "\nSession Start To Detect Test Cases\n");
    int correct = 0;
    for (int i = 0; i < sess->test_data_num; ++i)
    {
        load_test_data(sess, i);
        load_test_label(sess, i);
        forward_session(sess);
        Graph *graph = sess->graph;
        Layer **layers = graph->layers;
        if (sess->coretype == GPU){
            cudaMemcpy(sess->predicts, layers[graph->layer_num - 2]->output, \
                   layers[graph->layer_num - 2]->outputs*sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            memcpy(sess->predicts, layers[graph->layer_num - 2]->output, \
                    layers[graph->layer_num - 2]->outputs*sizeof(float));
        }
        test_information(sess->truth, sess->predicts, sess->label_num, sess->loss[0], sess->test_data_paths[i]);
        float max_pre = -1;
        int index = -1;
        for (int j = 0; j < sess->label_num; ++j){
            if (sess->predicts[j] > max_pre && sess->predicts[j] >= 0.5){
                max_pre = sess->predicts[j];
                index = j;
            }
        }
        if (index != -1 && sess->truth[index] == 1){
            correct += 1;
        }
    }
    fprintf(stderr, "\nTesting result: %d/%d  %f\n", correct, sess->test_data_num, (float)correct/(float)sess->test_data_num);
    fprintf(stderr, "\nSession Testing Finished\n\n");
}

void forward_session(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    Layer *l;
    float *input;
    if (sess->coretype == GPU){
        input = sess->input_gpu;
    } else {
        input = sess->input;
    }
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

void init_train_scene(Session *sess, char *weights_file)
{
    sess->input = calloc(sess->subdivision * sess->height * sess->width * sess->channel, sizeof(float));
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->input_gpu, \
               sess->subdivision*sess->height*sess->width*sess->channel*sizeof(float));
    }
    init_graph(sess->graph, sess->width, sess->height, sess->channel);
    get_workspace_size(sess);
    statistics_memory_occupy_size(sess);
    get_normalize_size(sess);
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
    create_predicts_memory(sess);
    set_truth_memory(sess);
    set_maxpool_index_memory(sess);
    set_dropout_rand_memory(sess);
    create_normalize_memory(sess);
    set_normalize_memory(sess);
    set_run_type(sess, 1);
}

void init_test_scene(Session *sess, char *weights_file)
{
    if (weights_file == NULL)
        return;
    sess->epoch = 1;
    sess->batch = 1;
    sess->subdivision = 1;
    sess->input = calloc(sess->subdivision * sess->height * sess->width * sess->channel, sizeof(float));
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->input_gpu, \
               sess->subdivision*sess->height*sess->width*sess->channel*sizeof(float));
    }
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
    create_predicts_memory(sess);
    set_truth_memory(sess);
    set_maxpool_index_memory(sess);
    set_dropout_rand_memory(sess);
    set_run_type(sess, 0);
}

void set_run_type(Session *sess, int train)
{
    Graph *graph = sess->graph;
    Layer *l = NULL;
    for (int i = 0; i < graph->layer_num; ++i){
        l = graph->layers[i];
        l->train = train;
    }
}

void test_information(float *truth, float *predict, int label_num, float loss, char *data_path)
{
    fprintf(stderr, "Test Data Path: %s\n", data_path);
    fprintf(stderr, "Truth:       Predict:\n");
    for (int i = 0; i < label_num; ++i){
        printf("%f     %f\n", truth[i], predict[i]);
    }
    fprintf(stderr, "Loss:    %f\n\n", loss);
}
