#include "manager.h"

void create_run_memory(Session *sess)
{
    create_workspace_memory(sess);
    create_output_memory(sess);
    create_delta_memory(sess);
    create_maxpool_index_memory(sess);
    create_dropout_rand_memory(sess);
}

void create_workspace_memory(Session *sess)
{
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->workspace_gpu, sess->workspace_size*sizeof(float));
    }
    sess->workspace = calloc(sess->workspace_size, sizeof(float));
    fprintf(stderr, "Apply For The Commond Work Space\n");
}

void create_input_memory(Session *sess)
{
    int inputs = sess->subdivision * sess->height * sess->width * sess->channel;
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->input_gpu, inputs*sizeof(float));
    }
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
    sess->output_size = outputs * sess->subdivision;
    sess->output = calloc(outputs * sess->subdivision, sizeof(float));
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->output_gpu, (outputs*sess->subdivision)*sizeof(float));
    }
    fprintf(stderr, "Apply For Layers Output Data Space\n");
}

void create_weights_memory(Session *sess)
{
    sess->weights = calloc(sess->weights_size, sizeof(float));
    sess->update_weights = calloc(sess->weights_size, sizeof(float));
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->weights_gpu, sess->weights_size*sizeof(float));
        cudaMalloc((void**)&sess->update_weights_gpu, sess->weights_size*sizeof(float));
    }
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
    sess->delta_size = deltas * sess->subdivision;
    sess->layer_delta = calloc(deltas * sess->subdivision, sizeof(float));
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->layer_delta_gpu, (deltas*sess->subdivision)*sizeof(float));
    }
    fprintf(stderr, "APPly For Layers Delta Data Space\n");
}

void create_label_memory(Session *sess)
{
    sess->label = calloc(sess->label_num * sess->subdivision, sizeof(char *));
}

void create_loss_memory(Session *sess)
{
    sess->loss = calloc(1, sizeof(float));
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->loss_gpu, sizeof(float));
    }
}

void create_truth_memory(Session *sess)
{
    sess->truth = calloc(sess->label_num * sess->subdivision, sizeof(float));
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->truth_gpu, (sess->label_num*sess->subdivision)*sizeof(float));
    }
}

void create_predicts_memory(Session *sess)
{
    sess->predicts = calloc(sess->label_num, sizeof(float));
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
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->maxpool_index_gpu, max_indexes*sess->subdivision*sizeof(int));
    }
    fprintf(stderr, "APPly For MAX Pool Layers's MAX Pixel Index Space\n");
}

void create_dropout_rand_memory(Session *sess)
{
    Graph *graph = sess->graph;
    int rand_num = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = graph->layers[i];
        if (l->type == DROPOUT) rand_num += l->inputs;
    }
    if (rand_num == 0){
        sess->dropout_rand = NULL;
        return;
    }
    sess->dropout_rand = calloc(rand_num * sess->subdivision, sizeof(int));
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->dropout_rand_gpu, rand_num*sess->subdivision*sizeof(int));
    }
    fprintf(stderr, "APPly For Dropout Layers's Rand Space\n");
}

void create_normalize_memory(Session *sess)
{
    if (sess->x_norm_size == 0) return;
    sess->x_norm = calloc(sess->x_norm_size*sess->subdivision, sizeof(float));
    sess->mean = calloc(sess->variance_size*sess->subdivision, sizeof(float));
    sess->roll_mean = calloc(sess->variance_size*sess->subdivision, sizeof(float));
    sess->variance = calloc(sess->variance_size*sess->subdivision, sizeof(float));
    sess->roll_variance = calloc(sess->variance_size*sess->subdivision, sizeof(float));
    sess->normalize_x = calloc(sess->normalize_x_size*sess->subdivision, sizeof(float));
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->x_norm_gpu, sess->x_norm_size*sess->subdivision*sizeof(float));
        cudaMalloc((void**)&sess->mean_gpu, sess->variance_size*sess->subdivision*sizeof(float));
        cudaMalloc((void**)&sess->roll_mean_gpu, sess->variance_size*sess->subdivision*sizeof(float));
        cudaMalloc((void**)&sess->variance_gpu, sess->variance_size*sess->subdivision*sizeof(float));
        cudaMalloc((void**)&sess->roll_variance_gpu, sess->variance_size*sess->subdivision*sizeof(float));
        cudaMalloc((void**)&sess->normalize_x_gpu, sess->normalize_x_size*sess->subdivision*sizeof(float));
    }
}

void set_graph_memory(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    float *output;
    float *layer_delta;
    float *workspace;
    int *maxpool_index;
    if (sess->coretype == GPU){
        output = sess->output_gpu;
        layer_delta = sess->layer_delta_gpu;
        workspace = sess->workspace_gpu;
        maxpool_index = sess->maxpool_index_gpu;  
    } else {
        output = sess->output;
        layer_delta = sess->layer_delta;
        workspace = sess->workspace;
        maxpool_index = sess->maxpool_index;
    }
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
    float *weights_g;
    float *update_weights_g;
    weights = sess->weights;
    update_weights = sess->update_weights;
    weights_g = sess->weights_gpu;
    update_weights_g = sess->update_weights_gpu;
    int weights_offset = 0;
    for (int i = 0; i < graph->layer_num; ++i)
    {
        Layer *l = layers[i];
        if (l->weights)
        {
            l->kernel_weights = weights + weights_offset;
            l->update_kernel_weights = update_weights + weights_offset;
            l->kernel_weights_gpu = weights_g + weights_offset;
            l->update_kernel_weights_gpu = update_weights_g + weights_offset;
            weights_offset += l->kernel_weights_size;
            if (l->bias || l->batchnorm)
            {
                l->bias_weights = weights + weights_offset;
                l->update_bias_weights = update_weights + weights_offset;
                l->bias_weights_gpu = weights_g + weights_offset;
                l->update_bias_weights_gpu = update_weights_g + weights_offset;
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
    if (sess->coretype == GPU){
        l->truth = sess->truth_gpu;  
    } else {
        l->truth = sess->truth;
    }
}

void set_maxpool_index_memory(Session *sess)
{
    if (sess->maxpool_index == NULL)
        return;
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    int *maxpool_index;
    if (sess->coretype == GPU){
        maxpool_index = sess->maxpool_index_gpu;
    } else {
        maxpool_index = sess->maxpool_index;
    }
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

void set_dropout_rand_memory(Session *sess)
{
    if (sess->dropout_rand == NULL) return;
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    int *dropout_rand;
    if (sess->coretype == GPU){
        dropout_rand = sess->dropout_rand_gpu;
    } else {
        dropout_rand = sess->dropout_rand;
    }
    int rand_offset = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = layers[i];
        if (l->type == DROPOUT){
            l->dropout_rand = dropout_rand + rand_offset;
            rand_offset += l->inputs * sess->subdivision;
        } else {
            l->dropout_rand = NULL;
        }
    }
    fprintf(stderr, "\nDropout Layers's Rand To Each Layer\n");
}

void set_normalize_memory(Session *sess)
{
    if (sess->x_norm_size == 0) return;
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    int x_norm_offset = 0;
    int variance_offset = 0;
    float *x_norm = sess->x_norm;
    float *mean = sess->mean;
    float *roll_mean = sess->roll_mean;
    float *variance = sess->variance;
    float *roll_variance = sess->roll_variance;
    float *normalize_x = sess->normalize_x;
    if (sess->coretype == GPU){
        x_norm = sess->x_norm_gpu;
        mean = sess->mean_gpu;
        roll_mean = sess->roll_mean_gpu;
        variance = sess->variance_gpu;
        roll_variance = sess->roll_variance_gpu;
        normalize_x = sess->normalize_x_gpu;
    }
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = layers[i];
        if (l->batchnorm){
            l->mean = mean + variance_offset;
            l->rolling_mean = roll_mean + variance_offset;
            l->variance = variance + variance_offset;
            l->rolling_variance = roll_variance + variance_offset;
            l->x_norm = x_norm + x_norm_offset;
            l->normalize_x = normalize_x + x_norm_offset;
            variance_offset += l->output_c * sess->subdivision;
            x_norm_offset += l->outputs * sess->subdivision;
        }
    }
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

void get_normalize_size(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    int x_norm_size = 0;
    int variance_size = 0;
    int normalize_x_size = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = layers[i];
        if (l->batchnorm){
            variance_size += l->output_c;
            x_norm_size += l->outputs;
            normalize_x_size += l->outputs;
        }
    }
    sess->x_norm_size = x_norm_size;
    sess->variance_size = variance_size;
    sess->normalize_x_size = normalize_x_size;
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
                Initializer init = sess->w_init;
                if (0 == strcmp(init.type, "val_init")){
                    val_init(l, init.val);
                } else if (0 == strcmp(init.type, "uniform_init")){
                    uniform_init(l, init.mean, init.variance);
                } else if (0 == strcmp(init.type, "normal_init")){
                    normal_init(l, init.mean, init.variance);
                } else if (0 == strcmp(init.type, "xavier_uniform_init")){
                    xavier_uniform_init(l);
                } else if (0 == strcmp(init.type, "xavier_normal_init")){
                    xavier_normal_init(l);
                } else if (0 == strcmp(init.type, "kaiming_uniform_init")){
                    kaiming_uniform_init(l, init.mode);
                } else if (0 == strcmp(init.type, "kaiming_normal_init")){
                    kaiming_normal_init(l, init.mode);
                } else if (0 == strcmp(init.type, "he_init")){
                    he_init(l);
                } else {
                    fprintf(stderr, "\nInitializer Error: no such kind of nInitializer\n");
                    return ;
                }
            }
            if (l->bias){
                fill_cpu(l->bias_weights, l->bias_weights_size, 0.001, 1);
            }
        }
        fprintf(stderr, "\nInit Weights\n");
    }
    memcpy(sess->update_weights, sess->weights, sess->weights_size * sizeof(float));
    if (sess->coretype == GPU){
        cudaMemcpy(sess->weights_gpu, sess->weights, sess->weights_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(sess->update_weights_gpu, sess->weights_gpu, sess->weights_size*sizeof(float), cudaMemcpyDeviceToDevice);
    }
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
    if (sess->coretype == GPU){
        cudaMemcpy(sess->input_gpu, sess->input, \
              (sess->height*sess->width*sess->channel*num)*sizeof(float), cudaMemcpyHostToDevice);
    }
}

void load_train_label(Session *sess, int index, int num)
{
    for (int i = index; i < index + num; ++i)
    {
        float *truth = sess->truth + (i - index) * sess->label_num;
        char *label_path = sess->train_label_paths[i];
        strip(label_path, ' ');
        void **labels = load_label_txt(label_path);
        int *lindex = (int*)labels[0];
        char *tmp = (char*)labels[1];
        for (int j = 0; j < sess->label_num; ++j){
            truth[j] = (float)atoi(tmp+lindex[j+1]);
        }
        free(lindex);
        free(tmp);
        free(labels);
    }
    if (sess->coretype == GPU){
        cudaMemcpy(sess->truth_gpu, sess->truth, sess->label_num*num*sizeof(float), cudaMemcpyHostToDevice);
    }
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
    if (sess->coretype == GPU){
        cudaMemcpy(sess->input_gpu, sess->input, \
              (sess->height*sess->width*sess->channel)*sizeof(float), cudaMemcpyHostToDevice);
    }
    free(im);
}

void load_test_label(Session *sess, int index)
{
    float *truth = sess->truth;
    char *label_path = sess->test_label_paths[index];
    strip(label_path, ' ');
    if (-1 == access(label_path, F_OK)){
        fprintf(stderr, "\nerror: %s is not exist\n", label_path);
        abort();
    }
    void **labels = load_label_txt(label_path);
    int *lindex = (int*)labels[0];
    char *tmp = (char*)labels[1];
    for (int i = 0; i < lindex[0]; ++i){
        truth[i] = (float)atoi(tmp+lindex[i+1]);
    }
    free(lindex);
    free(labels);
}

void save_weigths(Session *sess, char *path)
{
    FILE *fp = fopen(path, "wb");
    if (sess->coretype == GPU){
        cudaMemcpy(sess->weights, sess->weights_gpu, sess->weights_size*sizeof(float), cudaMemcpyDeviceToHost);
    }
    bfput(fp, sess->weights, sess->weights_size);
    fclose(fp);
}

void load_weights(Session *sess, char *path)
{
    FILE *fp = fopen(path, "rb");
    bfget(fp, sess->weights, sess->weights_size);
    if (sess->coretype == GPU){
        cudaMemcpy(sess->weights_gpu, sess->weights, sess->weights_size*sizeof(float), cudaMemcpyHostToDevice);
    }
    fclose(fp);
}

Session *load_session_json(char *graph_path, char *coretype)
{
    Session *sess = NULL;
    cJSON *cjson_graph = NULL;
    cJSON *cjson_public = NULL;
    cJSON *cjson_initializer = NULL;
    cJSON *cjson_layers = NULL;
    cJSON *cjson_width = NULL;
    cJSON *cjson_height = NULL;
    cJSON *cjson_channel = NULL;
    cJSON *cjson_batch = NULL;
    cJSON *cjson_subdivision = NULL;
    cJSON *cjson_epoch = NULL;
    cJSON *cjson_learning_rate = NULL;
    cJSON *cjson_label_num = NULL;
    Graph *graph = NULL;
    Initializer init = {0};
    int width = 0;
    int height = 0;
    int channel = 0;
    int batch = 0;
    int subdivision = 0;
    int epoch = 0;
    float learning_rate = 0;
    int label_num = 0;
    FILE *fp = fopen(graph_path, "r");
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *tmp = (char*)malloc(file_size * sizeof(char));
    memset(tmp, '\0', file_size * sizeof(char));
    fseek(fp, 0, SEEK_SET);
    fread(tmp, sizeof(char), file_size, fp);
    fclose(fp);
    cjson_graph = cJSON_Parse(tmp);
    cjson_public = cJSON_GetObjectItem(cjson_graph, "Public");
    cjson_initializer = cJSON_GetObjectItem(cjson_graph, "Initializer");
    cjson_layers = cJSON_GetObjectItem(cjson_graph, "Layers");
    cjson_width = cJSON_GetObjectItem(cjson_public, "width");
    cjson_height = cJSON_GetObjectItem(cjson_public, "height");
    cjson_channel = cJSON_GetObjectItem(cjson_public, "channel");
    cjson_batch = cJSON_GetObjectItem(cjson_public, "batch");
    cjson_subdivision = cJSON_GetObjectItem(cjson_public, "subdivision");
    cjson_epoch = cJSON_GetObjectItem(cjson_public, "epoch");
    cjson_learning_rate = cJSON_GetObjectItem(cjson_public, "learning_rate");
    cjson_label_num = cJSON_GetObjectItem(cjson_public, "label");
    init = load_initializer_json(cjson_initializer);
    graph = load_graph_json(cjson_layers);
    width = cjson_width->valueint;
    height = cjson_height->valueint;
    channel = cjson_channel->valueint;
    batch = cjson_batch->valueint;
    subdivision = cjson_subdivision->valueint;
    epoch = cjson_epoch->valueint;
    label_num = cjson_label_num->valueint;
    learning_rate = cjson_learning_rate->valuedouble;
    if (0 == strcmp(coretype, "gpu")){
        sess = create_session("gpu", init);
    } else {
        sess = create_session("cpu", init);
    }
    bind_graph(sess, graph);
    sess->height = height;
    sess->width = width;
    sess->channel = channel;
    sess->epoch = epoch;
    sess->batch = batch;
    sess->subdivision = subdivision;
    sess->learning_rate = learning_rate;
    sess->label_num = label_num;
    return sess;
}

Initializer load_initializer_json(cJSON *cjson_init)
{
    Initializer init = {0};
    cJSON *cjson_type = NULL;
    cJSON *cjson_mode = NULL;
    cJSON *cjson_mean = NULL;
    cJSON *cjson_variance = NULL;
    cJSON *cjson_val = NULL;
    char *type = NULL;
    char *mode = NULL;
    float mean = 0;
    float variance = 0;
    float val = 0;
    cjson_type = cJSON_GetObjectItem(cjson_init, "type");
    type = cjson_type->valuestring;
    if (0 == strcmp(type, "KN")){
        cjson_mode = cJSON_GetObjectItem(cjson_init, "mode");
        mode = cjson_mode->valuestring;
        init = kaiming_normal_initializer(mode);
    } else if (0 == strcmp(type, "KU")){
        cjson_mode = cJSON_GetObjectItem(cjson_init, "mode");
        mode = cjson_mode->valuestring;
        init = kaiming_uniform_initializer(mode);
    } else if (0 == strcmp(type, "XN")){
        init = xavier_normal_initializer();
    } else if (0 == strcmp(type, "XU")){
        init = xavier_uniform_initializer();
    } else if (0 == strcmp(type, "NM")){
        cjson_mean = cJSON_GetObjectItem(cjson_init, "mean");
        mean = cjson_mean->valuedouble;
        cjson_variance = cJSON_GetObjectItem(cjson_init, "variance");
        variance = cjson_variance->valuedouble;
        init = normal_initializer(mean, variance);
    } else if (0 == strcmp(type, "UM")){
        cjson_mean = cJSON_GetObjectItem(cjson_init, "mean");
        mean = cjson_mean->valuedouble;
        cjson_variance = cJSON_GetObjectItem(cjson_init, "variance");
        variance = cjson_variance->valuedouble;
        init = uniform_initializer(mean, variance);
    } else if (0 == strcmp(type, "VL")){
        cjson_val = cJSON_GetObjectItem(cjson_init, "val");
        val = cjson_val->valuedouble;
        init = val_initializer(val);
    } else if (0 == strcmp(type, "HE")){
        init = he_initializer();
    }
    return init;
}

Graph *load_graph_json(cJSON *cjson_graph)
{
    cJSON *cjson_layer = NULL;
    cJSON *cjson_type = NULL;
    cJSON *cjson_flag = NULL;
    cJSON *cjson_output = NULL;
    cJSON *cjson_bias = NULL;
    cJSON *cjson_active = NULL;
    cJSON *cjson_group = NULL;
    cJSON *cjson_ksize = NULL;
    cJSON *cjson_filters = NULL;
    cJSON *cjson_stride = NULL;
    cJSON *cjson_pad = NULL;
    cJSON *cjson_normalization = NULL;
    cJSON *cjson_probability = NULL;
    cJSON *cjson_shortcut_index = NULL;
    char *type = NULL;
    int ksize = 0;
    int output = 0;
    int bias = 0;
    char *active = NULL;
    int filters = 0;
    int stride = 0;
    int pad = 0;
    int normalization = 0;
    float probability = 0;
    int flag = 0;
    int group = 0;
    int shortcut_index = 0;
    int size = cJSON_GetArraySize(cjson_graph);
    Graph *graph = create_graph("Lumos", size);
    Layer *l = NULL;
    for (int i = 0; i < size; ++i){
        cjson_layer = cJSON_GetArrayItem(cjson_graph, i);
        cjson_type = cJSON_GetObjectItem(cjson_layer, "type");
        type = cjson_type->valuestring;
        if (0 == strcmp(type, "AVGPOOL")){
            cjson_ksize = cJSON_GetObjectItem(cjson_layer, "ksize");
            cjson_stride = cJSON_GetObjectItem(cjson_layer, "stride");
            cjson_pad = cJSON_GetObjectItem(cjson_layer, "pad");
            ksize = cjson_ksize->valueint;
            stride = cjson_stride->valueint;
            pad = cjson_pad->valueint;
            l = make_avgpool_layer(ksize, stride, pad);
        } else if (0 == strcmp(type, "CONNECT")){
            cjson_output = cJSON_GetObjectItem(cjson_layer, "output");
            cjson_bias = cJSON_GetObjectItem(cjson_layer, "bias");
            cjson_active = cJSON_GetObjectItem(cjson_layer, "active");
            output = cjson_output->valueint;
            bias = cjson_bias->valueint;
            active = cjson_active->valuestring;
            l = make_connect_layer(output, bias, active);
        } else if (0 == strcmp(type, "CONVOLUTIONAL")){
            cjson_filters = cJSON_GetObjectItem(cjson_layer, "filters");
            cjson_ksize = cJSON_GetObjectItem(cjson_layer, "ksize");
            cjson_stride = cJSON_GetObjectItem(cjson_layer, "stride");
            cjson_pad = cJSON_GetObjectItem(cjson_layer, "pad");
            cjson_bias = cJSON_GetObjectItem(cjson_layer, "bias");
            cjson_normalization = cJSON_GetObjectItem(cjson_layer, "normalization");
            cjson_active = cJSON_GetObjectItem(cjson_layer, "active");
            filters = cjson_filters->valueint;
            ksize = cjson_ksize->valueint;
            stride = cjson_stride->valueint;
            pad = cjson_pad->valueint;
            bias = cjson_bias->valueint;
            normalization = cjson_normalization->valueint;
            active = cjson_active->valuestring;
            l = make_convolutional_layer(filters, ksize, stride, pad, bias, normalization, active);
        } else if (0 == strcmp(type, "DROPOUT")){
            cjson_probability = cJSON_GetObjectItem(cjson_layer, "probability");
            probability = cjson_probability->valuedouble;
            l = make_dropout_layer(probability);
        } else if (0 == strcmp(type, "IM2COL")){
            cjson_flag = cJSON_GetObjectItem(cjson_layer, "flag");
            flag = cjson_flag->valueint;
            l = make_im2col_layer(flag);
        } else if (0 == strcmp(type, "MAXPOOL")){
            cjson_ksize = cJSON_GetObjectItem(cjson_layer, "ksize");
            cjson_stride = cJSON_GetObjectItem(cjson_layer, "stride");
            cjson_pad = cJSON_GetObjectItem(cjson_layer, "pad");
            ksize = cjson_ksize->valueint;
            stride = cjson_stride->valueint;
            pad = cjson_pad->valueint;
            l = make_maxpool_layer(ksize, stride, pad);
        } else if (0 == strcmp(type, "SOFTMAX")){
            cjson_group = cJSON_GetObjectItem(cjson_layer, "group");
            group = cjson_group->valueint;
            l = make_softmax_layer(group);
        } else if (0 == strcmp(type, "MSE")){
            cjson_group = cJSON_GetObjectItem(cjson_layer, "group");
            group = cjson_group->valueint;
            l = make_mse_layer(group);
        } else if (0 == strcmp(type, "SHORTCUT")){
            cjson_shortcut_index = cJSON_GetObjectItem(cjson_layer, "from");
            cjson_active = cJSON_GetObjectItem(cjson_layer, "active");
            shortcut_index = cjson_shortcut_index->valueint;
            active = cjson_active->valuestring;
            l = make_shortcut_layer(shortcut_index, active);
        }
        append_layer2grpah(graph, l);
    }
    return graph;
}
