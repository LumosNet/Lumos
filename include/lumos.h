#ifndef LUMOS_H
#define LUMOS_H

#include <stdio.h>
#include <stdlib.h>

#define CPU 0
#define GPU 1

typedef struct session Session;
typedef struct graph Graph;
typedef struct layer Layer;
typedef struct node Node;

typedef enum {
    CONVOLUTIONAL, CONNECT, MAXPOOL, AVGPOOL, \
    DROPOUT, MSE, SOFTMAX, SHORTCUT, NORMALIZE
} LayerType;

typedef enum {
    STAIR,
    HARDTAN,
    LINEAR,
    LOGISTIC,
    LOGGY,
    RELU,
    ELU,
    SELU,
    RELIE,
    RAMP,
    LEAKY,
    TANH,
    PLSE,
    LHTAN
} Activation;

typedef struct layer Layer;

typedef void (*forward)  (struct layer, int);
typedef void (*backward) (struct layer, float, int, float*);
typedef void (*update) (struct layer);
typedef forward Forward;
typedef backward Backward;
typedef update Update;

typedef void (*forward_gpu)  (struct layer, int);
typedef void (*backward_gpu) (struct layer, float, int, float*);
typedef void (*update_gpu) (struct layer);
typedef forward_gpu ForwardGpu;
typedef backward_gpu BackwardGpu;
typedef update_gpu UpdateGpu;

typedef void (*initialize) (struct layer *, int, int, int, int);
typedef void (*initialize_gpu) (struct layer *, int, int, int, int);
typedef initialize Initialize;
typedef initialize_gpu InitializeGpu;

typedef void (*weightinit) (struct layer, FILE*);
typedef weightinit WeightInit;
typedef void (*weightinit_gpu) (struct layer, FILE*);
typedef weightinit_gpu WeightInitGpu;

typedef void (*saveweights) (struct layer, FILE*);
typedef saveweights SaveWeights;
typedef void (*saveweights_gpu) (struct layer, FILE*);
typedef saveweights_gpu SaveWeightsGpu;

typedef void (*free_layer) (struct layer);
typedef free_layer FreeLayer;
typedef void (*free_layer_gpu) (struct layer);
typedef free_layer_gpu FreeLayerGpu;

struct layer{
    LayerType type;
    int status;
    int input_h;
    int input_w;
    int input_c;
    int output_h;
    int output_w;
    int output_c;

    int inputs;
    int outputs;

    int workspace_size;
    int truth_num;

    float *input;
    float *output;
    float *delta;
    float *truth;
    float *loss;
    float *workspace;

    int *maxpool_index;
    //为社么是指针
    int *dropout_rand;

    int filters;
    int ksize;
    int stride;
    int pad;
    int group;

    int bias;
    int normalize;
    // dropout 占比
    float probability;

    Layer *shortcut;
    int shortcut_index;

    float *kernel_weights;
    float *bias_weights;

    float *update_kernel_weights;
    float *update_bias_weights;

    float *mean;
    float *variance;
    float *rolling_mean;
    float *rolling_variance;
    float *x_norm;
    float *normalize_x;
    float *mean_delta;
    float *variance_delta;

    float *bn_scale;
    float *bn_bias;
    float *update_bn_scale;
    float *update_bn_bias;

    Forward forward;
    Backward backward;
    Update update;

    ForwardGpu forwardgpu;
    BackwardGpu backwardgpu;
    UpdateGpu updategpu;

    Initialize initialize;
    InitializeGpu initializegpu;

    WeightInit weightinit;
    WeightInitGpu weightinitgpu;

    Activation active;
    SaveWeights saveweights;
    SaveWeightsGpu saveweightsgpu;

    FreeLayer freelayer;
    FreeLayerGpu freelayergpu;
};

typedef struct graph{
    int status;
    float *input;
    float *output;
    float *delta;
    Node *head;
    Node *tail;
} graph, Graph;

struct node{
    Layer *l;
    Node *head;
    Node *next;
};

struct session{
    Graph *graph;

    int coretype;
    int epoch;
    int batch;
    int subdivision;

    int width;
    int height;
    int channel;

    float *loss;

    float learning_rate;
    size_t workspace_size;

    float *workspace;
    float *input;

    float *truth;
    int truth_num;

    int train_data_num;
    char **train_data_paths;
    char **train_label_paths;

    char *weights_path;
};

Session *create_session(Graph *graph, int h, int w, int c, int truth_num, char *type, char *path);
void init_session(Session *sess, char *data_path, char *label_path);
void set_train_params(Session *sess, int epoch, int batch, int subdivision, float learning_rate);
void set_detect_params(Session *sess);
void train(Session *sess, int binary);
void detect_classification(Session *sess, int binary);
void lr_scheduler_step(Session *sess, int step_size, float gamma);
void lr_scheduler_multistep(Session *sess, int *milestones, int num, float gamma);
void lr_scheduler_exponential(Session *sess, float gamma);
void lr_scheduler_cosineannealing(Session *sess, int T_max, float lr_min);

Graph *create_graph();
void append_layer2grpah(Graph *graph, Layer *l);

Layer *make_avgpool_layer(int ksize, int stride, int pad);
Layer *make_connect_layer(int output, int bias, char *active);
Layer *make_convolutional_layer(int filters, int ksize, int stride, int pad, int bias, int normalize, char *active);
Layer *make_dropout_layer(float probability);
Layer *make_maxpool_layer(int ksize, int stride, int pad);
Layer *make_softmax_layer(int group);

Layer *make_mse_layer(int group);
Layer *make_mae_layer(int group);
Layer *make_ce_layer(int group);

void init_constant(Layer *l, float x);
void init_normal(Layer *l, float mean, float std);
void init_uniform(Layer *l, float min, float max);
void init_kaiming_normal(Layer *l, float a, char *mode, char *nonlinearity);
void init_kaiming_uniform(Layer *l, float a, char *mode, char *nonlinearity);

#endif