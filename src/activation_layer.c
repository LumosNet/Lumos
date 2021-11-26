#include "activation_layer.h"

Layer *make_activation_layer(LayerParams *p, int h, int w, int c)
{
    Layer *layer = NULL;
    char *loss_name = NULL;
    if (0 == strcmp(p->type, "activation")){
        Layer *l = malloc(sizeof(Layer));
        l->type = ACTIVATION;
        l->input_h = h,
        l->input_w = w;
        l->input_c = c;
        Node *n = p->head;
        while (n){
            Params *param = n->val;
            if (0 == strcmp(param->key, "loss")){
                loss_name = param->val;
                l->loss = load_loss_type(param->val);
                l->forward = load_forward_loss(l->loss);
                l->backward = load_backward_loss(l->loss);
            } else if (0 == strcmp(param->key, "theta")){
                l->theta = atof(param->val);
            } else if (0 == strcmp(param->key, "gamma")){
                l->gamma = atof(param->val);
            }
            n = n->next;
        }
        l->output_h = 1;
        l->output_w = 1;
        l->output_c = 1;
        layer = l;
    }
    fprintf(stderr, "  loss    %s        %4d x%4d         ->     %d\n", \
            loss_name, 1, h*w*c, 1);
    return layer;
}