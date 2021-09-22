#include "network.h"

Network *load_network(char **cfg)
{
    NetParams *p = load_data_cfg("./cfg/lumos.cfg");
    Node *n = p->head;

    LayerParams 
    while (n){
        LayerParams *l = n->val;
        printf("%s\n", l->type);
        Node *N = l->head;
        while (N){
            Params *pa = N->val;
            printf("%s %s\n", pa->key, pa->val);
            N = N->next;
        }
        n = n->next;
    }
}