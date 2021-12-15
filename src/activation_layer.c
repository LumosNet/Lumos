#include "activation_layer.h"

// 行向量
Layer make_activation_layer(LayerParams *p, int batch, int h, int w, int c)
{
    Layer l = {0};
    char *loss_name = NULL;
    l.type = ACTIVATION;
    l.input_h = h,
    l.input_w = w;
    l.input_c = c;
    Node *n = p->head;
    while (n){
        Params *param = n->val;
        if (0 == strcmp(param->key, "loss")){
            loss_name = param->val;
            l.loss = load_loss_type(param->val);
            l.forward = load_forward_loss(l.loss);
            l.backward = load_backward_loss(l.loss);
        } else if (0 == strcmp(param->key, "theta")){
            l.theta = atof(param->val);
        } else if (0 == strcmp(param->key, "gamma")){
            l.gamma = atof(param->val);
        }
        n = n->next;
    }
    l.output_h = 1;
    l.output_w = 1;
    l.output_c = 1;

    if (l.loss == MSE) l.workspace_size = l.input_c*l.input_h*l.input_w + 1;
    else l.workspace_size = 0;

    int size_o[] = {l.output_w, l.output_h, l.output_c};
    int size_d[] = {l.input_w, l.input_h, l.input_c};
    l.output = malloc(batch*sizeof(struct Tensor *));
    l.delta = malloc(batch*sizeof(struct Tensor *));
    for (int i = 0; i < batch; ++i){
        l.output[i] = tensor_x(3, size_o, 0);
        l.delta[i] = tensor_x(3, size_d, 0);
    }

    fprintf(stderr, "  loss    %s        %4d x%4d         ->     %d\n", \
            loss_name, 1, h*w*c, 1);
    return l;
}