#include "parser.h"

NetParams *load_data_cfg(char *filecfg)
{
    FILE *fp = fopen(filecfg, "r");
    if (fp == 0) printf("文件打开失败!\n");
    NetParams *NP = make_net_params();
    LayerParams *LP;
    char *line;
    while ((line = fgetl(fp)) != 0)
    {
        if (line[0] == '\0') continue;
        strip(line);
        switch (line[0]){
            case '[':
                LP = make_layer_params(line);
                insert_net_params(NP, LP);
                break;
            case '#':
            default:
                insert_layer_params(LP, line);
                break;
        }
    }
    return NP;
}

NetParams *make_net_params()
{
    NetParams *NP = malloc(sizeof(NetParams));
    NP->size = 0;
    NP->head = NULL;
    NP->tail = NULL;
    return NP;
}

LayerParams *make_layer_params(char *line)
{
    int len = strlen(line);
    line = &line[1];
    line[len - 2] = '\0';
    LayerParams *LP = malloc(sizeof(LayerParams));
    LP->size = 0;
    LP->type = line;
    LP->head = NULL;
    LP->tail = NULL;
    return LP;
}

void insert_net_params(NetParams *NP, LayerParams *LP)
{
    Node *n = malloc(sizeof(Node));
    n->val = LP;
    if (NP->size == 0){
        n->prev = NULL;
        n->next = NULL;
        NP->head = n;
        NP->tail = n;
        NP->size += 1;
    }
    else{
        n->prev = NP->tail;
        n->next = NULL;
        NP->tail->next = n;
        NP->tail = n;
        NP->size += 1;
    }
}

void insert_layer_params(LayerParams *LP, char *line)
{
    Node *n = make_node_param(line);
    if (LP->size == 0){
        n->prev = NULL;
        n->next = NULL;
        LP->head = n;
        LP->tail = n;
        LP->size += 1;
    }
    else{
        n->prev = LP->tail;
        n->next = NULL;
        LP->tail->next = n;
        LP->tail = n;
        LP->size += 1;
    }
}

Node *make_node_param(char *line)
{
    int len = strlen(line);
    int i;
    Params *p = malloc(sizeof(Params));
    Node *n = malloc(sizeof(Node));
    char *a = 0;
    for (i = 0; i < len; ++i){
        if (line[i] == '='){
            line[i] = '\0';
            a = line + i + 1;
            break;
        }
    }
    p->key = line;
    p->val = a;
    n->val = p;
    return n;
}

Label *get_labels(char *path)
{
    FILE *fp = fopen(path, "r");
    char *line;
    Label *head = NULL;
    Label *tail = NULL;
    int *num = malloc(sizeof(int));
    while ((line = fgetl(fp)) != 0){
        if (line[0] == '\0') continue;
        Label *A = malloc(sizeof(Label *));
        if (tail) tail->next = A;
        char **sline = split(line, ' ', num);
        float *data = malloc(num[0]*sizeof(float));
        for (int i = 0; i < num[0]; ++i){
            data[i] = atof(sline[i]);
        }
        A->data = data;
        A->next = NULL;
        A->num = num[0];
        tail = A;
        if (head == NULL) head = A;
    }
    fclose(fp);
    return head;
}

char **read_lines(char *path, int *num)
{
    FILE *fp = fopen(path, "r");
    char *line;
    FLines *head = NULL;
    FLines *tail = NULL;
    int n = 0;
    while ((line = fgetl(fp)) != 0){
        if (line[0] == '\0') continue;
        FLines *L = malloc(sizeof(struct FLines *));
        if (tail) tail->next = L;
        L->data = line;
        tail = L;
        if (head == NULL) head = L;
        n += 1;
    }
    num[0] = n;
    char **lines = malloc(n*sizeof(char *));
    n = 0;
    while (head){
        lines[n] = head->data;
        head = head->next;
        n += 1;
    }
    return lines;
}