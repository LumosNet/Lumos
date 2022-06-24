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

void bind_test_data(Session *sess, char *path);

void bind_label(Session *sess, int label_num, char *path)
{
    FILE *fp = fopen(path, "r");
    char **label_paths = fgetls(fp);
    fclose(fp);
    sess->label_paths = label_paths+1;
    sess->label_num = label_num;
    fprintf(stderr, "\nGet Label List From %s\n", path);
}

void init_weights(Session *sess, char *weights_file)
{
    sess->weights = calloc(sess->weights_size, sizeof(float));
    sess->update_weights = calloc(sess->weights_size, sizeof(float));
    if (weights_file){
        FILE *fp = fopen(weights_file, "rb");
        bfget(fp, sess->weights, sess->weights_size);
        fprintf(stderr, "\nInit Weights From Weights File: %s\n", weights_file);
    } else{
        fprintf(stderr, "\nInit Weights\n");
    }
    memcpy(sess->weights, sess->update_weights, sess->weights_size*sizeof(float));
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


// 从index读取num个数据
void load_data(Session *sess, int index, int num)
{
    int h[1], w[1], c[1];
    float *im;
    int input_offset = 0;
    for (int i = index; i < index+num; ++i){
        char *data_path = sess->train_data_paths[i];
        im = load_image_data(data_path, w, h, c);
        resize_im(im, h[0], w[0], c[0], sess->height, sess->width, sess->input+input_offset);
        input_offset += sess->height*sess->width*sess->channel;
        free(im);
    }
    // fprintf(stderr, "Load Subdivision Data Succeed\n");
}

void load_label(Session *sess, int index, int num)
{
    int label_offset = 0;
    for (int i = index; i < index+num; ++i){
        FILE *fp = fopen(sess->label_paths[i], "r");
        char **label_paths = fgetls(fp);
        fclose(fp);
        int lines = atoi(label_paths[0]);
        for (int j = 0; j < lines; ++j){
            char *line = label_paths[j+1];
            int num[1];
            char **nodes = split(line, ' ', num);
            for (int k = 0; k < num[0]; ++k){
                strip(nodes[k], ' ');
                sess->label[label_offset+k] = nodes[k];
                label_offset += 1;
            }
        }
    }
}

void save_weigths(Session sess, char *path);
void load_weights(Session sess, char *path);
