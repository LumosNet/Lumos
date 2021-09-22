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