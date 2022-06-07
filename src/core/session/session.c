#include "session.h"

void bind_graph(Session sess, Graph graph)
{
    sess.graph = graph;
}

void bind_train_data(Session sess, char *path)
{
    FILE *fp = fopen(path, "r");
    char **data_paths = fgetls(fp);
}

void bind_test_data(Session sess, char *path);
// 从index读取num个数据
void load_data(Session sess, int index, int num);

void save_weigths(Session sess, char *path);
void load_weights(Session sess, char *path);
