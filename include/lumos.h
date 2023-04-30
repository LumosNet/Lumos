#ifndef LUMOS_H
#define LUMOS_H

#include <stdio.h>
#include <stdlib.h>

#include "cJSON.h"
#include "cJSON_Utils.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct initializer Initializer;
typedef struct layer Layer;

typedef void (*forward)  (struct layer, int);
typedef void (*backward) (struct layer, float, int, float*);
typedef forward Forward;
typedef backward Backward;

typedef void (*update) (struct layer, float, int, float*);
typedef update Update;

typedef int (*get_float_calculate_times) (struct layer*);
typedef get_float_calculate_times GetFloatCalculateTimes;

typedef float   (*activate)(float);
typedef float   (*gradient)(float);
typedef activate Activate;
typedef gradient Gradient;

typedef void (*label2truth) (char **, float *);
typedef label2truth Label2Truth;

typedef void (*process_test_information) (char **, float *, float *, float, char *);
typedef process_test_information ProcessTestInformation;

typedef enum {
    CONVOLUTIONAL, ACTIVATION, CONNECT, IM2COL, MAXPOOL, AVGPOOL, \
    MSE
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

struct initializer{
    char *type;
    float val;
    float mean;
    float variance;
    char *mode;
};

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

    Activation active;
    Activation gradient;

    Update update;
    GetFloatCalculateTimes get_fct;
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
    Initializer w_init;
} Session;

void val_init(Layer *l, float val);
void uniform_init(Layer *l, float mean, float variance);
void normal_init(Layer *l, float mean, float variance);
void xavier_uniform_init(Layer *l);
void xavier_normal_init(Layer *l);
void kaiming_uniform_init(Layer *l, char *mode);
void kaiming_normal_init(Layer *l, char *mode);
void he_init(Layer *l);

Initializer val_initializer(float val);
Initializer uniform_initializer(float mean, float variance);
Initializer normal_initializer(float mean, float variance);
Initializer xavier_uniform_initializer();
Initializer xavier_normal_initializer();
Initializer kaiming_uniform_initializer(char *mode);
Initializer kaiming_normal_initializer(char *mode);
Initializer he_initializer();

Graph *create_graph(char *name, int layer_n);

void append_layer2grpah(Graph *graph, Layer *l);
void init_graph(Graph *g, int w, int h, int c);

Activation load_activate_type(char *activate);

Activate load_activate(Activation TYPE);
Gradient load_gradient(Activation TYPE);

float activate_x(Activation TYPE, float x);
float gradient_x(Activation TYPE, float x);
void activate_list(float *origin, int num, Activation TYPE);
void gradient_list(float *origin, int num, Activation TYPE);

Layer *make_avgpool_layer(int ksize);

void init_avgpool_layer(Layer *l, int w, int h, int c);

void forward_avgpool_layer(Layer l, int num);
void backward_avgpool_layer(Layer l, float rate, int num, float *n_delta);

Layer *make_connect_layer(int output, int bias, char *active);
void init_connect_layer(Layer *l, int w, int h, int c);

void forward_connect_layer(Layer l, int num);
void backward_connect_layer(Layer l, float rate, int num, float *n_delta);

void update_connect_layer(Layer l, float rate, int num, float *n_delta);

Layer *make_convolutional_layer(int filters, int ksize, int stride, int pad, int bias, int normalization, char *active);
void init_convolutional_layer(Layer *l, int w, int h, int c);

void forward_convolutional_layer(Layer l, int num);
void backward_convolutional_layer(Layer l, float rate, int num, float *n_delta);

void update_convolutional_layer(Layer l, float rate, int num, float *n_delta);

Layer *make_im2col_layer(int flag);

void init_im2col_layer(Layer *l, int w, int h, int c);

void forward_im2col_layer(Layer l, int num);
void backward_im2col_layer(Layer l, float rate, int num, float *n_delta);

Layer *make_maxpool_layer(int ksize);

void init_maxpool_layer(Layer *l, int w, int h, int c);

void forward_maxpool_layer(Layer l, int num);
void backward_maxpool_layer(Layer l, float rate, int num, float *n_delta);

Layer *make_mse_layer(int group);

void init_mse_layer(Layer *l, int w, int h, int c);

void forward_mse_layer(Layer l, int num);
void backward_mse_layer(Layer l, float rate, int num, float *n_delta);

Layer *make_softmax_layer(int group);

void init_softmax_layer(Layer *l, int w, int h, int c);

void forward_softmax_layer(Layer l, int num);
void backward_softmax_layer(Layer l, float rate, int num, float *n_delta);

Layer *make_dropout_layer(float probability);

void init_dropout_layer(Layer *l, int w, int h, int c);

void forward_dropout_layer(Layer l, int num);
void backward_dropout_layer(Layer l, float rate, int num, float *n_delta);

void add_bias(float *origin, float *bias, int n, int size);
void fill_cpu(float *data, int len, float x, int offset);
void multy_cpu(float *data, int len, float x, int offset);
void add_cpu(float *data, int len, float x, int offset);

void min_cpu(float *data, int num, float *space);
void max_cpu(float *data, int num, float *space);
void sum_cpu(float *data, int num, float *space);
void mean_cpu(float *data, int num, float *space);

void matrix_add_cpu(float *data_a, float *data_b, int num, float *space);
void matrix_subtract_cpu(float *data_a, float *data_b, int num, float *space);
void matrix_multiply_cpu(float *data_a, float *data_b, int num, float *space);
void matrix_divide_cpu(float *data_a, float *data_b, int num, float *space);

void saxpy_cpu(float *data_a, float *data_b, int num, float x, float *space);
void sum_channel_cpu(float *data, int h, int w, int c, float ALPHA, float *space);

void one_hot_encoding(int n, int label, float *space);
void gemm(int TA, int TB, int AM, int AN, int BM, int BN, float ALPHA, 
        float *A, float *B, float *C);

void gemm_nn(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C);
void gemm_tn(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C);
void gemm_nt(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C);
void gemm_tt(int AM, int AN, int BM, int BN, float ALPHA, float *A, float *B, float *C);
void im2col(float *img, int h, int w, int c, int ksize, int stride, int pad, float *space);
void col2im(float *img, int ksize, int stride, int pad, int out_h, int out_w, int out_c, float *space);

// 统计图像中不同灰度等级的像素点数量
int *census_image_pixel(float *img, int w, int h, int c);
// 统计通道中不同灰度等级的像素点数量
int *census_channel_pixel(float *img, int w, int h, int c, int index_c);
float *load_image_data(char *img_path, int *w, int *h, int *c);
void save_image_data(float *img, int w, int h, int c, char *savepath);
// 双线性内插值
void resize_im(float *img, int height, int width, int channel, int row, int col, float *space);

void avgpool(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space);
void maxpool(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space, int *index);

void avgpool_gradient(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n);
void maxpool_gradient(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n, int *index);

float uniform_data(float a, float b, int *seed);
float guass_data(float mean, float sigma, int *seed);

void uniform_list(float a, float b, int num, float *space);
void guass_list(float mean, float sigma, int seed, int num, float *space);
void normal_list(int num, float *space);

float rand_normal();
float rand_uniform(float min, float max);

float rand_normal();

void session_train(Session *sess, char *weights_path);
void session_test(Session *sess);

void forward_session(Session *sess);
void backward_session(Session *sess);

void init_train_scene(Session *sess, char *weights_file);
void init_test_scene(Session *sess, char *weights_file);

void create_run_memory(Session *sess);
void create_workspace_memory(Session *sess);
void create_input_memory(Session *sess);
void create_output_memory(Session *sess);
void create_weights_memory(Session *sess);
void create_delta_memory(Session *sess);
void create_label_memory(Session *sess);
void create_loss_memory(Session *sess);
void create_truth_memory(Session *sess);
void create_predicts_memory(Session *sess);
void create_maxpool_index_memory(Session *sess);

void set_graph_memory(Session *sess);
void set_graph_weight(Session *sess);
void set_label(Session *sess);
void set_loss_memory(Session *sess);
void set_truth_memory(Session *sess);
void set_maxpool_index_memory(Session *sess);

void get_workspace_size(Session *sess);
void statistics_memory_occupy_size(Session *sess);

void init_weights(Session *sess, char *weights_file);

// 从index读取num个数据
void load_train_data(Session *sess, int index, int num);
void load_train_label(Session *sess, int index, int num);

void load_test_data(Session *sess, int index);
void load_test_label(Session *sess, int index);

void save_weigths(Session *sess, char *path);
void load_weights(Session *sess, char *path);

Session *create_session(char *type, Initializer w_init);

void bind_graph(Session *sess, Graph *graph);
void bind_train_data(Session *sess, char *path);
void bind_test_data(Session *sess, char *path);
void bind_train_label(Session *sess, char *path);
void bind_test_label(Session *sess, char *path);
void bind_label2truth_func(Session *sess, int truth_num, Label2Truth func);

void set_input_dimension(Session *sess, int h, int w, int c);
void set_train_params(Session *sess, int epoch, int batch, int subdivision, float learning_rate);

void strip(char *line, char c);
void padding_string(char *space, char *str, int index);

char *inten2str(int x);
char *int2str(int x);

Session *load_session_json(char *graph_path, char *coretype);
Initializer load_initializer_json(cJSON *cjson_init);
Graph *load_graph_json(cJSON *cjson_graph);

void test_information(float *truth, float *predict, float loss, char *data_path);

#ifdef __cplusplus
}
#endif

#endif