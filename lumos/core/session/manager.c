#include "manager.h"

void create_run_memory(Session *sess)
{
    create_workspace_memory(sess);
    create_output_memory(sess);
    create_delta_memory(sess);
    create_maxpool_index_memory(sess);
}

void create_workspace_memory(Session *sess)
{
#ifdef GPU
    cudaMalloc((void**)&sess->workspace_gpu, sess->workspace_size*sizeof(float));
#endif
    sess->workspace = calloc(sess->workspace_size, sizeof(float));
    fprintf(stderr, "Apply For The Commond Work Space\n");
}

void create_input_memory(Session *sess)
{
    int inputs = sess->subdivision * sess->height * sess->width * sess->channel;
#ifdef GPU
    cudaMalloc((void**)&sess->input_gpu, inputs*sizeof(float));
#endif
    sess->input = calloc(inputs, sizeof(float));
}

void create_output_memory(Session *sess)
{
    Graph *graph = sess->graph;
    int outputs = 0;
    for (int i = 0; i < graph->layer_num; ++i)
    {
        Layer *l = graph->layers[i];
        outputs += l->outputs;
    }
    sess->output = calloc(outputs * sess->subdivision, sizeof(float));
#ifdef GPU
    cudaMalloc((void**)&sess->output_gpu, (outputs*sess->subdivision)*sizeof(float));
#endif
    fprintf(stderr, "Apply For Layers Output Data Space\n");
}

void create_weights_memory(Session *sess)
{
    sess->weights = calloc(sess->weights_size, sizeof(float));
    sess->update_weights = calloc(sess->weights_size, sizeof(float));
#ifdef GPU
    cudaMalloc((void**)&sess->weights_gpu, sess->weights_size*sizeof(float));
    cudaMalloc((void**)&sess->update_weights_gpu, sess->weights_size*sizeof(float));
#endif
    fprintf(stderr, "Apply For Graph Weights Space\n");
}

void create_delta_memory(Session *sess)
{
    Graph *graph = sess->graph;
    int deltas = 0;
    for (int i = 0; i < graph->layer_num; ++i)
    {
        Layer *l = graph->layers[i];
        deltas += l->deltas;
    }
    sess->layer_delta = calloc(deltas * sess->subdivision, sizeof(float));
#ifdef GPU
    cudaMalloc((void**)&sess->layer_delta_gpu, (deltas*sess->subdivision)*sizeof(float));
#endif
    fprintf(stderr, "APPly For Layers Delta Data Space\n");
}

void create_label_memory(Session *sess)
{
    sess->label = calloc(sess->label_num * sess->subdivision, sizeof(char *));
}

void create_loss_memory(Session *sess)
{
    sess->loss = calloc(1, sizeof(float));
#ifdef GPU
    cudaMalloc((void**)&sess->loss_gpu, sizeof(float));
#endif
}

void create_truth_memory(Session *sess)
{
    sess->truth = calloc(sess->truth_num * sess->subdivision, sizeof(float));
#ifdef GPU
    cudaMalloc((void**)&sess->truth_gpu, (sess->truth_num*sess->subdivision)*sizeof(float));
#endif
}

void create_predicts_memory(Session *sess)
{
    sess->predicts = calloc(sess->truth_num, sizeof(float));
}

void create_maxpool_index_memory(Session *sess)
{
    Graph *graph = sess->graph;
    int max_indexes = 0;
    for (int i = 0; i < graph->layer_num; ++i)
    {
        Layer *l = graph->layers[i];
        if (l->type == MAXPOOL)
            max_indexes += l->outputs;
    }
    if (max_indexes == 0)
    {
        sess->maxpool_index = NULL;
        return;
    }
    sess->maxpool_index = calloc(max_indexes * sess->subdivision, sizeof(int));
#ifdef GPU
    cudaMalloc((void**)&sess->maxpool_index_gpu, max_indexes*sess->subdivision*sizeof(int));
#endif
    fprintf(stderr, "APPly For MAX Pool Layers's MAX Pixel Index Space\n");
}

void set_graph_memory(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    float *output;
    float *layer_delta;
    float *workspace;
    int *maxpool_index;
#ifdef GPU
    output = sess->output_gpu;
    layer_delta = sess->layer_delta_gpu;
    workspace = sess->workspace_gpu;
    maxpool_index = sess->maxpool_index_gpu;
#else
    output = sess->output;
    layer_delta = sess->layer_delta;
    workspace = sess->workspace;
    maxpool_index = sess->maxpool_index;
#endif
    int offset_o = 0;
    int delta_offset = 0;
    for (int i = 0; i < graph->layer_num; ++i)
    {
        Layer *l = layers[i];
        l->output = output + offset_o;
        l->delta = layer_delta + delta_offset;
        l->workspace = workspace;
        if (l->type == MAXPOOL)
        {
            l->maxpool_index = maxpool_index + offset_o;
        }
        offset_o += l->outputs * sess->subdivision;
        delta_offset += l->deltas * sess->subdivision;
    }
    fprintf(stderr, "\nDistribut Running Memory To Each Layer\n");
}

void set_graph_weight(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    float *weights;
    float *update_weights;
#ifdef GPU
    weights = sess->weights_gpu;
    update_weights = sess->update_weights_gpu;
#else
    weights = sess->weights;
    update_weights = sess->update_weights;
#endif
    int weights_offset = 0;
    for (int i = 0; i < graph->layer_num; ++i)
    {
        Layer *l = layers[i];
        if (l->weights)
        {
            l->kernel_weights = weights + weights_offset;
            l->update_kernel_weights = update_weights + weights_offset;
            weights_offset += l->kernel_weights_size;
            if (l->bias)
            {
                l->bias_weights = weights + weights_offset;
                l->update_bias_weights = update_weights + weights_offset;
                weights_offset += l->bias_weights_size;
            }
        }
    }
    fprintf(stderr, "\nDistribut Weights To Each Layer\n");
}

void set_label(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    for (int i = 0; i < graph->layer_num; ++i)
    {
        Layer *l = layers[i];
        l->label = sess->label;
    }
}

void set_loss_memory(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    Layer *l = layers[graph->layer_num - 1];
    l->loss = sess->loss;
}

void set_truth_memory(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    Layer *l = layers[graph->layer_num - 1];
#ifdef GPU
    l->truth = sess->truth_gpu;
#else
    l->truth = sess->truth;
#endif
}

void set_maxpool_index_memory(Session *sess)
{
    if (sess->maxpool_index == NULL)
        return;
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    int *maxpool_index;
#ifdef GPU
    maxpool_index = sess->maxpool_index_gpu;
#else
    maxpool_index = sess->maxpool_index;
#endif
    int index_offset = 0;
    for (int i = 0; i < graph->layer_num; ++i)
    {
        Layer *l = layers[i];
        if (l->type == MAXPOOL)
        {
            l->maxpool_index = maxpool_index + index_offset;
            index_offset += l->outputs * sess->subdivision;
        }
        else
        {
            l->maxpool_index = NULL;
        }
    }
    fprintf(stderr, "\nDistribut MAX Pool Layers's MAX Pixel Index To Each Layer\n");
}

void get_workspace_size(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    int max_workspace_size = -1;
    int weights_size = 0;
    for (int i = 0; i < graph->layer_num; ++i)
    {
        Layer *l = layers[i];
        if (l->workspace_size > max_workspace_size)
        {
            max_workspace_size = l->workspace_size;
        }
        if (l->weights)
        {
            weights_size += l->kernel_weights_size;
            weights_size += l->bias_weights_size;
        }
    }
    sess->workspace_size = max_workspace_size;
    sess->weights_size = weights_size;
}

void statistics_memory_occupy_size(Session *sess)
{
    Graph *graph = sess->graph;
    int outputs = 0;
    int deltas = 0;
    int weights = 0;
    int max_indexes = 0;
    for (int i = 0; i < graph->layer_num; ++i)
    {
        Layer *l = graph->layers[i];
        if (l->type == MAXPOOL)
            max_indexes += l->outputs;
        outputs += l->outputs;
        deltas += l->deltas;
        weights += l->kernel_weights_size;
        weights += l->bias_weights_size;
    }
    sess->memory_size += outputs * sess->subdivision * sizeof(float);
    sess->memory_size += deltas * sess->subdivision * sizeof(float);
    sess->memory_size += max_indexes * sess->subdivision * sizeof(float);
    sess->memory_size += sess->workspace_size * sizeof(float);
    sess->memory_size += sess->subdivision * sess->height * sess->width * sess->channel * sizeof(float);
    float mem_size = (float)sess->memory_size / 1024 / 1024 / 1024;
    if (mem_size > 0.1)
    {
        fprintf(stderr, "\nThe Whole Training Process Will Occupy %.1fGB Host Memory\n", mem_size);
    }
    else
    {
        mem_size = (float)sess->memory_size / 1024 / 1024;
        fprintf(stderr, "\nThe Whole Training Process Will Occupy %.1fMB Host Memory\n", mem_size);
    }
    fprintf(stderr, "Please Make Sure The Host Have Enough Memory\n\n");
}

void init_weights(Session *sess, char *weights_file)
{
    if (weights_file)
    {
        load_weights(sess, weights_file);
        fprintf(stderr, "\nInit Weights From Weights File: %s\n", weights_file);
    }
    else
    {
        Graph *graph = sess->graph;
        Layer **layers = graph->layers;
        for (int i = 0; i < graph->layer_num; ++i)
        {
            Layer *l = layers[i];
            if (l->weights){
                l->init_layer_weights(l);
            }
        }
        fprintf(stderr, "\nInit Weights\n");
    }
    memcpy(sess->update_weights, sess->weights, sess->weights_size * sizeof(float));
#ifdef GPU
    cudaMemcpy(sess->update_weights_gpu, sess->weights_gpu, sess->weights_size*sizeof(float), cudaMemcpyDeviceToDevice);
#endif
}

void load_train_data(Session *sess, int index, int num)
{
    int h[1], w[1], c[1];
    float *im;
    int offset_i = 0;
    for (int i = index; i < index + num; ++i)
    {
        char *data_path = sess->train_data_paths[i];
        strip(data_path, ' ');
        im = load_image_data(data_path, w, h, c);
        resize_im(im, h[0], w[0], c[0], sess->height, sess->width, sess->input + offset_i);
        offset_i += sess->height * sess->width * sess->channel;
        free(im);
    }
#ifdef GPU
    cudaMemcpy(sess->input_gpu, sess->input, \
              (sess->height*sess->width*sess->channel*num)*sizeof(float), cudaMemcpyHostToDevice);
#endif
}

void load_train_label(Session *sess, int index, int num)
{
    for (int i = index; i < index + num; ++i)
    {
        float *truth = sess->truth + (i - index) * sess->truth_num;
        char *label_path = sess->train_label_paths[i];
        strip(label_path, ' ');
        char **label = load_label_txt(label_path, sess->label_num);
        sess->label2truth(label, truth);
    }
#ifdef GPU
    cudaMemcpy(sess->truth_gpu, sess->truth, sess->truth_num*num*sizeof(float), cudaMemcpyHostToDevice);
#endif
}

void load_test_data(Session *sess, int index)
{
    int h[1], w[1], c[1];
    float *im;
    char *data_path = sess->test_data_paths[index];
    strip(data_path, ' ');
    if (-1 == access(data_path, F_OK)){
        fprintf(stderr, "\nerror: %s is not exist\n", data_path);
        abort();
    }
    im = load_image_data(data_path, w, h, c);
    resize_im(im, h[0], w[0], c[0], sess->height, sess->width, sess->input);
#ifdef GPU
    cudaMemcpy(sess->input_gpu, sess->input, \
              (sess->height*sess->width*sess->channel)*sizeof(float), cudaMemcpyHostToDevice);
#endif
    free(im);
}

char **load_test_label(Session *sess, int index)
{
    float *truth = sess->truth;
    char *label_path = sess->test_label_paths[index];
    strip(label_path, ' ');
    if (-1 == access(label_path, F_OK)){
        fprintf(stderr, "\nerror: %s is not exist\n", label_path);
        abort();
    }
    char **label = load_label_txt(label_path, sess->label_num);
    sess->label2truth(label, truth);
    return label;
}

void save_weigths(Session *sess, char *path)
{
    FILE *fp = fopen(path, "wb");
#ifdef GPU
    cudaMemcpy(sess->weights, sess->weights_gpu, sess->weights_size*sizeof(float), cudaMemcpyDeviceToHost);
#endif
    bfput(fp, sess->weights, sess->weights_size);
    fclose(fp);
}

void load_weights(Session *sess, char *path)
{
    FILE *fp = fopen(path, "rb");
    bfget(fp, sess->weights, sess->weights_size);
#ifdef GPU
    cudaMemcpy(sess->weights_gpu, sess->weights, sess->weights_size*sizeof(float), cudaMemcpyHostToDevice);
#endif
    fclose(fp);
}
