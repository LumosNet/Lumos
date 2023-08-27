#include "session.h"

void bind_train_data(Session *sess, char *path)
{
    char *tmp = fget(path);
    int num = format_str_line(tmp);
    int offset = 0;
    for (int i = 0; i < num; ++i){
        sess->dataset_pathes[i] = tmp + offset;
        offset += strlen(tmp)+1;
    }
}

void bind_train_label(Session *sess, char *path)
{
    char *tmp = fget(path);
    int num = format_str_line(tmp);
    int offset = 0;
    for (int i = 0; i < num; ++i){
        sess->labelset_pathes[i] = tmp + offset;
        offset += strlen(tmp)+1;
    }
}

void create_workspace(Session *sess)
{
    int max = -TMP_MAX;
    Graph *graph = sess->graph;
    for (int i = 0; i < graph->num; ++i){
        Layer *l = graph->layers[i];
        if (max <= l->workspace_size) max = l->workspace_size;
    }
    sess->workspace = calloc(max, sizeof(float));
}

int count_running_memsize(Session *sess)
{
    return 0;
}
