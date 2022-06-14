#include "session.h"

Session *create_session()
{
    Session *sess = malloc(sizeof(Session));
    return sess;
}

void del_session()
{

}

void bind_graph(Session *sess, Graph graph)
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
}

void bind_test_data(Session *sess, char *path);

void set_input_dimension(Session sess, int h, int w, int c)
{
    sess.height = h;
    sess.width = w;
    sess.channel = c;
}

void set_train_params(Session sess, int epoch, int batch, int subdivision, float learning_rate)
{
    sess.epoch = epoch;
    sess.batch = batch;
    sess.subdivision = subdivision;
    sess.learning_rate = learning_rate;
}

// 从index读取num个数据
void load_data(Session sess, int index, int num)
{
    char **data_paths = sess.train_data_paths;
    int h[1], w[1], c[1];
    float *im;
    int input_offset = 0;
    for (int i = index; i < index+num; ++i){
        char *data_path = sess.train_data_paths[i];
        im = load_image_data(data_path, w, h, c);
        resize_im(im, h[0], w[0], c[0], sess.height, sess.width, sess.input+input_offset);
        input_offset += sess.height*sess.width*sess.channel;
        free(im);
    }
}

void save_weigths(Session sess, char *path);
void load_weights(Session sess, char *path);
