#ifndef LUMOS_H
#define LUMOS_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct layer Layer;

typedef void (*forward)  (struct layer, int);
typedef void (*backward) (struct layer, float, int, float*);
typedef forward Forward;
typedef backward Backward;

typedef void (*update) (struct layer, float, int, float*);
typedef update Update;

typedef void (*init_layer_weights) (struct layer*);
typedef init_layer_weights InitLayerWeights;

typedef void (*label2truth) (char **, float *);
typedef label2truth Label2Truth;

typedef void (*process_test_information) (char **, float *, float *, float, char *);
typedef process_test_information ProcessTestInformation;

typedef float   (*activate)(float);
typedef float   (*gradient)(float);
typedef activate Activate;
typedef gradient Gradient;

typedef struct CFGParam{
    char *key;
    char *val;
    struct CFGParam *next;
} CFGParam;

typedef struct CFGParams{
    struct CFGParam *head;
    struct CFGParam *tail;
} CFGParams;

typedef struct CFGPiece{
    int param_num;
    char *name;
    struct CFGParams *params;
    struct CFGPiece *next;
} CFGPiece;

typedef struct CFGPieces{
    struct CFGPiece *head;
    struct CFGPiece *tail;
} CFGPieces;

typedef struct CFG{
    int piece_num;
    struct CFGPieces *pieces;
} CFG;

typedef enum {
    CONVOLUTIONAL, ACTIVATION, CONNECT, IM2COL, MAXPOOL, AVGPOOL, \
    MSE
} LayerType;

struct layer{
    LayerType type;
    int input_h;
    int input_w;
    int input_c;
    int output_h;
    int output_w;
    int output_c;

    int inputs;
    int outputs;
    int kernel_weights_size;
    int bias_weights_size;
    int deltas;
    int workspace_size;
    int label_num;

    char *active_str;

    float *input;
    float *output;
    float *delta;
    char **label;
    float *truth;
    float *loss;

    float *workspace;

    int *maxpool_index;

    int filters;
    int ksize;
    int stride;
    int pad;
    int group;
    int im2col_flag;

    int weights;
    int bias;
    int batchnorm;

    // 在网中的位置，0开始
    int index;
    // 浮点数操作数
    int fops;

    float *kernel_weights;
    float *bias_weights;

    float *update_kernel_weights;
    float *update_bias_weights;

    Forward forward;
    Backward backward;

    Activate active;
    Gradient gradient;

    Update update;

    InitLayerWeights init_layer_weights;
};

typedef struct graph{
    char *graph_name;
    int layer_list_num;
    int layer_num;
    int height;
    int channel;
    float *input;
    float *output;
    float *delta;
    Layer **layers;

    int kinds;
    char **label;

    int num;
    char **data;

    CFG *cfg;
} graph, Graph;

typedef struct session{
    Graph *graph;

    int epoch;
    int batch;
    int subdivision;

    int width;
    int height;
    int channel;

    float *loss;

    float learning_rate;
    size_t workspace_size;
    size_t weights_size;

    float *workspace;
    float *input;
    float *output;
    float *layer_delta;

    char **label;
    int label_num;

    float *truth;
    int truth_num;

    float *predicts;

    int *maxpool_index;

    float *weights;
    float *update_weights;

    int train_data_num;
    char **train_data_paths;
    char **train_label_paths;

    int test_data_num;
    char **test_data_paths;
    char **test_label_paths;

    int memory_size;

    Label2Truth label2truth;
} Session;

Graph *create_graph(char *name, int layer_n);
Graph *create_graph_by_cfg(CFGPiece *p, int layer_n);
Graph *load_graph_from_cfg(char *cfg_path);

void append_layer2grpah(Graph *graph, Layer *l);

void init_graph(Graph *g, int w, int h, int c);

Layer *make_avgpool_layer(int ksize);
Layer *make_avgpool_layer_by_cfg(CFGParams *p);

void init_avgpool_layer(Layer *l, int w, int h, int c);

void forward_avgpool_layer(Layer l, int num);
void backward_avgpool_layer(Layer l, float rate, int num, float *n_delta);

Layer *make_connect_layer(int output, int bias, char *active);
Layer *make_connect_layer_by_cfg(CFGParams *p);

void init_connect_layer(Layer *l, int w, int h, int c);
void init_connect_weights(Layer *l);

void forward_connect_layer(Layer l, int num);
void backward_connect_layer(Layer l, float rate, int num, float *n_delta);

void update_connect_layer(Layer l, float rate, int num, float *n_delta);

Layer *make_convolutional_layer(int filters, int ksize, int stride, int pad, int bias, int normalization, char *active);
Layer *make_convolutional_layer_by_cfg(CFGParams *p);

void init_convolutional_layer(Layer *l, int w, int h, int c);
void init_convolutional_weights(Layer *l);

void forward_convolutional_layer(Layer l, int num);
void backward_convolutional_layer(Layer l, float rate, int num, float *n_delta);

void update_convolutional_layer(Layer l, float rate, int num, float *n_delta);
Layer *make_im2col_layer(int flag);
Layer *make_im2col_layer_by_cfg(CFGParams *p);

void init_im2col_layer(Layer *l, int w, int h, int c);

void forward_im2col_layer(Layer l, int num);
void backward_im2col_layer(Layer l, float rate, int num, float *n_delta);

Layer *make_maxpool_layer(int ksize);
Layer *make_maxpool_layer_by_cfg(CFGParams *p);

void init_maxpool_layer(Layer *l, int w, int h, int c);

void forward_maxpool_layer(Layer l, int num);
void backward_maxpool_layer(Layer l, float rate, int num, float *n_delta);

Layer *make_mse_layer(int group);
Layer *make_mse_layer_by_cfg(CFGParams *p);

void init_mse_layer(Layer *l, int w, int h, int c);

void forward_mse_layer(Layer l, int num);
void backward_mse_layer(Layer l, float rate, int num, float *n_delta);

void bind_graph(Session *sess, Graph *graph);

void session_train(Session *sess, float learning_rate, char *weights_path);
void session_test(Session *sess, ProcessTestInformation process_test_information);

void create_train_scene(Session *sess, int h, int w, int c, int label_num, int truth_num, Label2Truth func, char *dataset_list_file, char *label_list_file);
void init_train_scene(Session *sess, int epoch, int batch, int subdivision, char *weights_file);

void create_test_scene(Session *sess, int h, int w, int c, int label_num, int truth_num, Label2Truth func, char *dataset_list_file, char *label_list_file);
void init_test_scene(Session *sess, char *weights_file);

Session *create_session();
void del_session();

void one_hot_encoding(int n, int label, float *space);

#ifdef __cplusplus
}
#endif

#endif