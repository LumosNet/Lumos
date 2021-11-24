#include "softmax_layer.h"

void forward_softmax_layer(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *input = l->input[i];
        float *ezk = malloc(l->group*sizeof(float));
        float sum = 0;
        for (int j = 0; j < l->group; ++j){
            ezk[j] = pow(M_E, input->data[j]);
            sum += ezk[j];
        }
        Tensor *output = l->output[i];
        for (int j = 0; j < l->group; ++j){
            output->data[j] = ezk[j] / sum;
        }
    }
}

void backward_softmax_layer(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *input = l->input[i];
        Tensor *delta = net->delta[i];
        float *ezk = malloc(l->group*sizeof(float));
        float sum = 0;
        for (int j = 0; j < l->group; ++j){
            ezk[j] = pow(M_E, input->data[j]);
            sum += ezk[j];
        }
        int size[] = {l->group, l->group};
        Tensor *gradient_soft = tensor_x(2, size, 0);
        for (int j = 0; j < l->group; ++j){
            for (int k = 0; k < l->group; ++k){
                if (j == k){
                    gradient_soft->data[j*l->group + k] = ezk[j]/sum - ezk[j]*ezk[j];
                } else{
                    gradient_soft->data[j*l->group + k] = -(ezk[j]*ezk[k]);
                }
            }
        }
        net->delta[i] = gemm(delta, gradient_soft);
    }
}

Layer *make_softmax_layer(LayerParams *p, int h, int w, int c)
{
    Layer *layer = NULL;
    if (0 == strcmp(p->type, "softmax")){
        Layer *l = malloc(sizeof(Layer));
        l->type = SOFTMAX;
        l->input_h = h;
        l->input_w = w;
        l->input_c = c;
        Node *n = p->head;
        while (n){
            Params *param = n->val;
            if (0 == strcmp(param->key, "group")){
                l->group = atoi(param->val);
            }
            n = n->next;
        }
        l->output_h = l->input_h;
        l->output_w = l->input_w;
        l->output_c = l->input_c;
        l->forward = forward_softmax_layer;
        l->backward = backward_softmax_layer;
        layer = l;
    }
    fprintf(stderr, "  softmax          %5d      %4d x%4d         ->  %4d x%4d\n", \
            layer->group, layer->input_h, layer->input_w, layer->output_h, layer->output_w);
    return layer;
}