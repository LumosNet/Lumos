#include "session.h"

Session *create_session()
{
    Session *sess = malloc(sizeof(Session));
    sess->memory_size = 0;
    return sess;
}

void bind_graph(Session *sess, Graph *graph)
{
    sess->graph = graph;
}

void bind_train_data(Session *sess, char *path)
{
    FILE *fp = fopen(path, "r");
    char **data_paths = fgetls(fp);
    fclose(fp);
    int lines = atoi(data_paths[0]);
    sess->train_data_num = lines;
    sess->train_data_paths = data_paths+1;
    fprintf(stderr, "\nGet Train Data List From %s\n", path);
}

void bind_test_data(Session *sess, char *path)
{
    FILE *fp = fopen(path, "r");
    char **data_paths = fgetls(fp);
    fclose(fp);
    int lines = atoi(data_paths[0]);
    sess->test_data_num = lines;
    sess->test_data_paths = data_paths+1;
    fprintf(stderr, "\nGet Test Data List From %s\n", path);
}

void bind_label(Session *sess, int label_num, char *path)
{
    FILE *fp = fopen(path, "r");
    char **label_paths = fgetls(fp);
    fclose(fp);
    sess->train_label_paths = label_paths+1;
    sess->label_num = label_num;
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = layers[i];
        l->label_num = label_num;
    }
    fprintf(stderr, "\nGet Label List From %s\n", path);
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
