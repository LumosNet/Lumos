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

void create_weights_memory(Session *sess)
{
    sess->weights = calloc(sess->weights_size, sizeof(float));
    sess->update_weights = calloc(sess->weights_size, sizeof(float));
    fprintf(stderr, "Apply For Graph Weights Space\n");
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

void create_label_memory(Session *sess)
{
    sess->label = calloc(sess->label_num*sess->subdivision, sizeof(char*));
}

void create_loss_memory(Session *sess)
{
    sess->loss = calloc(1, sizeof(float));
}

void create_maxpool_index_memory(Session *sess)
{
    Graph *graph = sess->graph;
    int max_indexes = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = graph->layers[i];
        if (l->type == MAXPOOL) max_indexes += l->outputs;
    }
    if (max_indexes == 0){
        sess->maxpool_index = NULL;
        return;
    }
    sess->maxpool_index = calloc(max_indexes*sess->subdivision, sizeof(float));
    fprintf(stderr, "APPly For MAX Pool Layers's MAX Pixel Index Space\n");
}

void set_graph_memory(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    int offset_o = 0;
    int delta_offset = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = layers[i];
        l->output = sess->output+offset_o;
        l->delta = sess->layer_delta+delta_offset;
        l->workspace = sess->workspace;
        offset_o += l->outputs*sess->subdivision;
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

void set_label(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = layers[i];
        l->label = sess->label;
    }
}

void set_loss_memory(Session *sess)
{
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    Layer *l = layers[graph->layer_num-1];
    l->loss = sess->loss;
}

void set_maxpool_index_memory(Session *sess)
{
    if (sess->maxpool_index == NULL) return;
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    int index_offset = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = layers[i];
        if (l->type == MAXPOOL){
            l->maxpool_index = sess->maxpool_index+index_offset;
            index_offset += l->outputs*sess->subdivision;
        } else{
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
}

void statistics_memory_occupy_size(Session *sess)
{
    Graph *graph = sess->graph;
    int outputs = 0;
    int deltas = 0;
    int max_indexes = 0;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = graph->layers[i];
        if (l->type == MAXPOOL) max_indexes += l->outputs;
        outputs += l->outputs;
        deltas += l->deltas;
    }
    sess->memory_size += outputs*sess->subdivision*sizeof(float);
    sess->memory_size += deltas*sess->subdivision*sizeof(float);
    sess->memory_size += max_indexes*sess->subdivision*sizeof(float);
    sess->memory_size += sess->workspace_size*sizeof(float);
    sess->memory_size += sess->subdivision*sess->height*sess->width*sess->channel*sizeof(float);
    float mem_size = (float)sess->memory_size / 1024 / 1024 / 1024;
    if (mem_size > 0.1){
        fprintf(stderr, "\nThe Whole Training Process Will Occupy %.1fGB Host Memory\n", mem_size);
    } else{
        mem_size = (float)sess->memory_size / 1024 / 1024;
        fprintf(stderr, "\nThe Whole Training Process Will Occupy %.1fMB Host Memory\n", mem_size);
    }
    fprintf(stderr, "Please Make Sure The Host Have Enough Memory\n\n");
}

void init_weights(Session *sess, char *weights_file)
{
    if (weights_file){
        load_weights(sess, weights_file);
        fprintf(stderr, "\nInit Weights From Weights File: %s\n", weights_file);
    } else{
        Graph *graph = sess->graph;
        Layer **layers = graph->layers;
        for (int i = 0; i < graph->layer_num; ++i){
            Layer *l = layers[i];
            if (l->init_layer_weights) l->init_layer_weights(l);
        }
        fprintf(stderr, "\nInit Weights\n");
    }
    memcpy(sess->update_weights, sess->weights, sess->weights_size*sizeof(float));
}


void load_data(Session *sess, int index, int num)
{
    int h[1], w[1], c[1];
    float *im;
    int offset_i = 0;
    for (int i = index; i < index+num; ++i){
        char *data_path = sess->train_data_paths[i];
        im = load_image_data(data_path, w, h, c);
        resize_im(im, h[0], w[0], c[0], sess->height, sess->width, sess->input+offset_i);
        offset_i += sess->height*sess->width*sess->channel;
        free(im);
    }
}

void load_label(Session *sess, int index, int num)
{
    for (int i = index; i < index+num; ++i){
        char **label = load_label_txt(sess->label_paths[i], sess->label_num);
        for (int j = 0; j < sess->label_num; ++j){
            sess->label[(i-index)*sess->label_num+j] = label[j];
        }
    }
    for (int i = 0; i < num; ++i){
    }
}

void save_weigths(Session *sess, char *path)
{
    FILE *fp = fopen(path, "wb");
    bfput(fp, sess->weights, sess->weights_size);
    fclose(fp);
}

void load_weights(Session *sess, char *path)
{
    FILE *fp = fopen(path, "rb");
    bfget(fp, sess->weights, sess->weights_size);
    fclose(fp);
}
