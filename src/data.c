#include "data.h"

void load_train_data(Network *net, int offset)
{
    int *w = malloc(sizeof(int));
    int *h = malloc(sizeof(int));
    int *c = malloc(sizeof(int));
    for (int i = 0; i < net->batch; ++i){
        int index = offset + i;
        if (index >= net->num) index -= net->num;
        float *im = load_image_data(net->data[index], w, h, c);
        if (w[0] != net->width || h[0] != net->height){
            float *new = calloc(net->height*net->width*c[0], sizeof(float));
            resize_im(im, h[0], w[0], c[0], net->height, net->width, new);
            free(im);
            im = new;
        }
        int offset_i = i*net->height*net->width*c[0];
        memcpy_float_list(net->input, im, offset_i, 0, net->height*net->width*net->channel);
        free(im);
        int *num = malloc(sizeof(int));
        int *n = malloc(sizeof(int));
        char **lines = read_lines(net->label[index], num);
        Label *head = NULL;
        Label *tail = NULL;
        for (int j = 0; j < num[0]; ++j){
            Label *A = malloc(sizeof(Label *));
            if (tail) tail->next = A;
            char **sline = split(lines[j], ' ', n);
            float *data = malloc(n[0]*sizeof(float));
            for (int j = 0; j < n[0]; ++j){
                data[j] = atof(sline[j]);
            }
            A->data = data;
            A->next = NULL;
            A->num = n[0];
            tail = A;
            if (head == NULL) {
                head = A;
            }
        }
        net->labels[i] = head[0];
    }
    net->output = net->input;
    free(w);
    free(h);
    free(c);
}

void load_train_path(Network *net, char *data_path, char *label_path)
{
    FILE *fp = fopen(data_path, "r");
    int i;
    int num = 0;
    char *line;
    while ((line = fgetl(fp)) != 0)
    {
        if (line[0] == '\0') continue;
        num += 1;
    }
    fclose(fp);
    fp = fopen(data_path, "r");
    net->data = malloc(num*sizeof(char *));
    net->label = malloc(num*sizeof(char *));
    net->num = num;
    i = 0;
    while ((line = fgetl(fp)) != 0)
    {
        if (line[0] == '\0') continue;
        net->data[i] = line;
        i += 1;
    }
    fclose(fp);
    fp = fopen(label_path, "r");
    i = 0;
    while ((line = fgetl(fp)) != 0)
    {
        if (line[0] == '\0') continue;
        net->label[i] = line;
        i += 1;
    }
    fclose(fp);
}

void load_weights(Network *net, char *file)
{
    FILE *fp = NULL;
    if (file) fp = fopen(file, "rb");
    for (int i = 0; i < net->n; ++i){
        Layer l = net->layers[i];
        if (l.lweights) l.lweights(l, fp);
    }
    if (file) fclose(fp);
}

void save_weights(Network *net, char *file)
{
    FILE *fp = fopen(file, "wb");
    for (int i = 0; i < net->n; ++i){
        Layer l = net->layers[i];
        if (l.sweights) l.sweights(l, fp);
    }
    fclose(fp);
}