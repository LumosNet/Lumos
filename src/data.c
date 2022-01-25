#include "data.h"

void load_train_data(Network *net, int offset)
{
    int *w = malloc(sizeof(int));
    int *h = malloc(sizeof(int));
    int *c = malloc(sizeof(int));
    for (int i = 0; i < net->batch; ++i){
        int offset_i = i*net->height*net->width*c[0];
        int index = offset + i;
        if (index >= net->num) index -= net->num;
        float *im = load_image_data(net->data[index], w, h, c);
        if (w[0] != net->width || h[0] != net->height){
            resize_im(im, h[0], w[0], c[0], net->height, net->width, net->input+offset_i);
        }
        net->labels[i] = get_labels(net->label[index])[0];
    }
    net->output = net->input;
    free(w);
    free(h);
    free(c);
}

void load_train_path(Network *net, char *data_path, char *label_path)
{
    printf("%s\n%s\n", data_path, label_path);
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
    FILE *fl = fopen(label_path, "r");
    i = 0;
    while ((line = fgetl(fl)) != 0)
    {
        if (line[0] == '\0') continue;
        net->label[i] = line;
        i += 1;
    }
    fclose(fl);
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