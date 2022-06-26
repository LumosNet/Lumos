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
    Graph *graph = sess->graph;
    Layer **layers = graph->layers;
    for (int i = 0; i < graph->layer_num; ++i){
        Layer *l = layers[i];
        l->label_num = label_num;
    }
    fprintf(stderr, "\nGet Label List From %s\n", path);
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
        printf("%s\n", data_path);
        im = load_image_data(data_path, w, h, c);
        resize_im(im, h[0], w[0], c[0], sess->height, sess->width, sess->input+input_offset);
        input_offset += sess->height*sess->width*sess->channel;
        free(im);
    }
}

void load_label(Session *sess, int index, int num)
{
    printf("start load label\n");
    int label_offset = 0;
    for (int i = index; i < index+num; ++i){
        FILE *fp = fopen(sess->label_paths[i], "r");
        printf("lable path: %s\n", sess->label_paths[i]);
        char **labels = fgetls(fp);
        fclose(fp);
        int lines = atoi(labels[0]);
        printf("lines: %d\n", lines);
        for (int j = 0; j < lines; ++j){
            char *line = labels[j+1];
            printf("line: %s\n", line);
            int num[1];
            char **nodes = split(line, ' ', num);
            for (int k = 0; k < num[0]; ++k){
                strip(nodes[k], ' ');
                sess->label[label_offset+k] = nodes[k];
                label_offset += 1;
            }
        }
    }
    printf("finif load label\n");
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
