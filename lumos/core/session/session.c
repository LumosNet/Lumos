#include "session.h"

Session *create_session(char *type, Initializer w_init)
{
    Session *sess = malloc(sizeof(Session));
    sess->memory_size = 0;
    sess->w_init = w_init;
    if (0 == strcmp(type, "gpu")){
        sess->coretype = GPU;
    } else {
        sess->coretype = CPU;
    }
    return sess;
}

void bind_graph(Session *sess, Graph *graph)
{
    sess->graph = graph;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = graph->layers[i];
        l->coretype = sess->coretype;
    }
}

void bind_train_data(Session *sess, char *path)
{
    char *tmp = fget(path);
    int *index = split(tmp, '\n');
    int lines = index[0];
    sess->train_data_num = lines;
    sess->train_data_paths = malloc(lines*sizeof(char*));
    for (int i = 0; i < lines; ++i){
        sess->train_data_paths[i] = tmp+index[i+1];
    }
    free(index);
    fprintf(stderr, "\nGet Train Data List From %s\n", path);
}

void bind_test_data(Session *sess, char *path)
{
    char *tmp = fget(path);
    int *index = split(tmp, '\n');
    int lines = index[0];
    sess->test_data_num = lines;
    sess->test_data_paths = malloc(lines*sizeof(char*));
    for (int i = 0; i < lines; ++i){
        sess->test_data_paths[i] = tmp+index[i+1];
    }
    free(index);
    fprintf(stderr, "\nGet Test Data List From %s\n", path);
}

void bind_train_label(Session *sess, char *path)
{
    char *tmp = fget(path);
    int *index = split(tmp, '\n');
    int lines = index[0];
    sess->train_label_paths = malloc(lines*sizeof(char*));
    for (int i = 0; i < lines; ++i){
        sess->train_label_paths[i] = tmp+index[i+1];
    }
    free(index);
    fprintf(stderr, "\nGet Label List From %s\n", path);
}

void bind_test_label(Session *sess, char *path)
{
    char *tmp = fget(path);
    int *index = split(tmp, '\n');
    int lines = index[0];
    sess->test_label_paths = malloc(lines*sizeof(char*));
    for (int i = 0; i < lines; ++i){
        sess->test_label_paths[i] = tmp+index[i+1];
    }
    free(index);
    fprintf(stderr, "\nGet Label List From %s\n", path);
}

void bind_label2truth_func(Session *sess, int truth_num, Label2Truth func)
{
    sess->label2truth = func;
    sess->truth_num = truth_num;
}

void set_input_dimension(Session *sess, int h, int w, int c)
{
    sess->height = h;
    sess->width = w;
    sess->channel = c;
    fprintf(stderr, "\nSet Input Dementions: Height Width Channel\n");
    fprintf(stderr, "                       %3d    %3d   %3d\n", h, w, c);
}

void set_train_params(Session *sess, int epoch, int batch, int subdivision, float learning_rate)
{
    sess->epoch = epoch;
    sess->batch = batch;
    sess->subdivision = subdivision;
    sess->learning_rate = learning_rate;
}
