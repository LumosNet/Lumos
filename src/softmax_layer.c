#include "softmax_layer.h"

void forward_softmax_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        float sum = 0;
        for (int j = 0; j < l.group; ++j){
            net.workspace[j] += pow(M_E, l.input[i]->data[j]);
            sum += net.workspace[j];
        }
        for (int j = 0; j < l.group; ++j){
            l.output[i]->data[j] = net.workspace[j] / sum;
        }
    }
    printf("softmax\n");
}

void backward_softmax_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        float sum = 0;
        for (int j = 0; j < l.group; ++j){
            net.workspace[j] = pow(M_E, l.input[i]->data[j]);
            sum += net.workspace[j];
        }
        for (int j = 0; j < l.group; ++j){
            for (int k = 0; k < l.group; ++k){
                if (j == k){
                    net.workspace[j*l.group + k] = net.workspace[j]/sum - net.workspace[j]*net.workspace[j];
                } else{
                    net.workspace[j*l.group + k] = -(net.workspace[j]*net.workspace[k]);
                }
            }
        }
        gemm(0, 0, net.delta[i]->size[1], net.delta[i]->size[0], l.group, l.group, 1, net.delta[i]->data, net.workspace, l.delta[i]->data);
    }
}

Layer make_softmax_layer(LayerParams *p, int batch, int h, int w, int c)
{
    Layer l = {0};
    l.type = SOFTMAX;
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;
    Node *n = p->head;
    while (n){
        Params *param = n->val;
        if (0 == strcmp(param->key, "group")){
            l.group = atoi(param->val);
        }
        n = n->next;
    }
    l.output_h = l.group;
    l.output_w = 1;
    l.output_c = 1;
    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;

    l.workspace_size = l.group*l.group;

    int size_o[] = {l.output_w, l.output_h, l.output_c};
    int size_d[] = {l.input_w, l.input_h, l.input_c};
    l.output = malloc(batch*sizeof(Tensor *));
    l.delta = malloc(batch*sizeof(Tensor *));
    for (int i = 0; i < batch; ++i){
        l.output[i] = tensor_x(3, size_o, 0);
        l.delta[i] = tensor_x(3, size_d, 0);
    }

    fprintf(stderr, "  softmax          %5d      %4d x%4d         ->  %4d x%4d\n", \
            l.group, l.input_h, l.input_w, l.output_h, l.output_w);
    return l;
}