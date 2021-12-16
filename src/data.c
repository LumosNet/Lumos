#include "data.h"

void load_train_data(Network *net, int offset)
{
    for (int i = 0; i < net->batch; ++i){
        int index = offset + i;
        if (index >= net->num) index -= net->num;
        Tensor *im = load_image_data(net->data[index]);
        if (im->size[0] != net->width || im->size[1] != net->height){
            printf("gan\n");
            float *new = calloc(net->height*net->width*im->size[2], sizeof(float));
            resize_im(im->data, im->size[1], im->size[0], im->size[2], 
                    net->height, net->width, new);
            free(im->data);
            im->data = new;
            im->size[0] = net->width;
            im->size[1] = net->height;
            im->num = net->height*net->width*im->size[2];
        }
        net->input[i] = im;
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