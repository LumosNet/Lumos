#include "manager.h"

Session *load_session_json(char *graph_path, char *coretype)
{
    Session *sess = malloc(sizeof(Session *));
    cJSON *cjson_graph = NULL;
    cJSON *cjson_public = NULL;
    cJSON *cjson_initializer = NULL;
    cJSON *cjson_layers = NULL;
    Initializer init = {0};
    char *tmp = fget(graph_path);
    cjson_graph = cJSON_Parse(tmp);
    cjson_public = cJSON_GetObjectItem(cjson_graph, "Public");
    cjson_initializer = cJSON_GetObjectItem(cjson_graph, "Initializer");
    cjson_layers = cJSON_GetObjectItem(cjson_graph, "Layers");
    sess->epoch = cJSON_GetObjectItem(cjson_public, "epoch")->valueint;
    sess->batch = cJSON_GetObjectItem(cjson_public, "batch")->valueint;
    sess->subdivison = cJSON_GetObjectItem(cjson_public, "subdivision")->valueint;
    sess->learning_rate = cJSON_GetObjectItem(cjson_public, "learning_rate")->valuedouble;
    sess->height = cJSON_GetObjectItem(cjson_public, "height")->valueint;
    sess->width = cJSON_GetObjectItem(cjson_public, "width")->valueint;
    sess->channel = cJSON_GetObjectItem(cjson_public, "channel")->valueint;
    sess->dataset_listf = cJSON_GetObjectItem(cjson_public, "dataset")->valuestring;
    sess->labelset_listf = cJSON_GetObjectItem(cjson_public, "labelset")->valuestring;
    sess->graph = load_graph_json(cjson_layers);
    sess->init = load_initializer_json(cjson_initializer);
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
    Graph *graph = malloc(sizeof(Graph));
    int num = cJSON_GetArraySize(cjson_graph);
    graph->num = num;
    Layer **layers = load_layers(cjson_graph);
    graph->layers = layers;
    return graph;
}

Layer **load_layers(cJSON *cjson_graph)
{
    cJSON *cjson_layer = NULL;
    cJSON *cjson_type = NULL;
    int num = cJSON_GetArraySize(cjson_graph);
    Layer **ret = malloc(num*sizeof(Layer*));
    for (int i = 0; i < num; ++i){
        cjson_layer = cJSON_GetArrayItem(cjson_graph, i);
        ret[i] = load_layer_json(cjson_layer);
    }
    return ret;
}

Layer *load_layer_json(cJSON *cjson_layer)
{
    Layer *l = NULL;
    char *type = cJSON_GetObjectItem(cjson_layer, "type")->valuestring;
    if (0 == strcmp(type, "AVGPOOL")){
        l = load_avgpool_layer_json(cjson_layer);
    } else if (0 == strcmp(type, "CONNECT")){
        l = load_connect_layer_json(cjson_layer);
    } else if (0 == strcmp(type, "CONVOLUTIONAL")){
        l = load_convolutional_layer_json(cjson_layer);
    } else if (0 == strcmp(type, "DROPOUT")){
        l = load_dropout_layer_json(cjson_layer);
    } else if (0 == strcmp(type, "IM2COL")){
        l = load_im2col_layer_json(cjson_layer);
    } else if (0 == strcmp(type, "MAXPOOL")){
        l = load_maxpool_layer_json(cjson_layer);
    } else if (0 == strcmp(type, "SOFTMAX")){
        l = load_softmax_layer_json(cjson_layer);
    } else if (0 == strcmp(type, "MSE")){
        l = load_mse_layer_json(cjson_layer);
    } else if (0 == strcmp(type, "SHORTCUT")){
        l = load_shortcut_layer_json(cjson_layer);
    } else {
        fprintf(stderr, "Layer Type ERROR: Unknown Type!");
        return NULL;
    }
    return l;
}

Layer *load_avgpool_layer_json(cJSON *cjson_layer)
{
    Layer *l;
    int ksize = cJSON_GetObjectItem(cjson_layer, "ksize")->valueint;
    int stride = cJSON_GetObjectItem(cjson_layer, "stride")->valueint;
    int pad = cJSON_GetObjectItem(cjson_layer, "pad")->valueint;
    l = make_avgpool_layer(ksize, stride, pad);
    return l;
}

Layer *load_connect_layer_json(cJSON *cjson_layer)
{
    Layer *l;
    int output = cJSON_GetObjectItem(cjson_layer, "output")->valueint;
    int bias = cJSON_GetObjectItem(cjson_layer, "bias")->valueint;
    int normalization = cJSON_GetObjectItem(cjson_layer, "normalization")->valueint;
    char *active = cJSON_GetObjectItem(cjson_layer, "active")->valuestring;
    l = make_connect_layer(output, bias, normalization, active);
    return l;
}

Layer *load_convolutional_layer_json(cJSON *cjson_layer)
{
    Layer *l;
    int filters = cJSON_GetObjectItem(cjson_layer, "filters")->valueint;
    int ksize = cJSON_GetObjectItem(cjson_layer, "ksize")->valueint;
    int stride = cJSON_GetObjectItem(cjson_layer, "stride")->valueint;
    int pad = cJSON_GetObjectItem(cjson_layer, "pad")->valueint;
    int bias = cJSON_GetObjectItem(cjson_layer, "bias")->valueint;
    int normalization = cJSON_GetObjectItem(cjson_layer, "normalization")->valueint;
    char *active = cJSON_GetObjectItem(cjson_layer, "active")->valuestring;
    l = make_convolutional_layer(filters, ksize, stride, pad, bias, normalization, active);
    return l;
}

Layer *load_dropout_layer_json(cJSON *cjson_layer)
{
    Layer *l;
    float probability = cJSON_GetObjectItem(cjson_layer, "probability")->valuedouble;
    l = make_dropout_layer(probability);
    return l;
}

Layer *load_im2col_layer_json(cJSON *cjson_layer)
{
    Layer *l;
    l = make_im2col_layer();
    return l;
}

Layer *load_maxpool_layer_json(cJSON *cjson_layer)
{
    Layer *l;
    int ksize = cJSON_GetObjectItem(cjson_layer, "ksize")->valueint;
    int stride = cJSON_GetObjectItem(cjson_layer, "stride")->valueint;
    int pad = cJSON_GetObjectItem(cjson_layer, "pad")->valueint;
    l = make_maxpool_layer(ksize, stride, pad);
    return l;
}

Layer *load_softmax_layer_json(cJSON *cjson_layer)
{
    Layer *l;
    int group = cJSON_GetObjectItem(cjson_layer, "group")->valueint;
    l = make_softmax_layer(group);
    return l;
}

Layer *load_mse_layer_json(cJSON *cjson_layer)
{
    Layer *l;
    int group = cJSON_GetObjectItem(cjson_layer, "group")->valueint;
    l = make_mse_layer(group);
    return l;
}

Layer *load_shortcut_layer_json(cJSON *cjson_layer)
{
    Layer *l;
    int shortcut_index = cJSON_GetObjectItem(cjson_layer, "from")->valueint;
    char *active = cJSON_GetObjectItem(cjson_layer, "active")->valuestring;
    l = make_shortcut_layer(shortcut_index, active);
    return l;
}

void train(Session *sess)
{
    for (int i = 0; i < sess->epoch; ++i){
        int sub_epochs = (int)(sess->train_data_num / sess->batch);
        int sub_batchs = (int)(sess->batch / sess->subdivision);
        if (sess->train_data_num % sess->batch != 0) sub_epochs += 1;
        if (sess->batch % sess->subdivision != 0){
            fprintf(stderr, "Running Error: Subdivision Error!");
            return ;
        }
        for (int j = 0; j < sub_epochs; ++j){
            for (int k = 0; k < sub_batchs; ++k){
                load_dataandlabel(sess);
                run_forward(sess);
                run_backward(sess);
                run_update(sess);
            }
        }
    }
}

void detect(Session *sess)
{
    sess->subdivison = 1;
    if (load_dataandlabel(sess) == 0) return ;
    run_forward(sess);
}

void run_forward(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    for (int i = 0; i < graph->num; ++i){
        Layer *l = layers[i];
        l->forward(l, sess->subdivison);
    }
}

void run_backward(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    for (int i = 0; i < graph->num; ++i){
        Layer *l = layers[i];
        l->backward(l, sess->learning_rate, sess->subdivison);
    }
}

void run_update(Session *sess)
{

}
