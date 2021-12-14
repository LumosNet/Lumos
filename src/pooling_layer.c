#include "pooling_layer.h"

PoolingType load_pooling_type(char *pool)
{
    if (0 == strcmp(pool, "average")) return AVG;
    return MAX;
}

void forward_pooling_layer(Layer *l, Network *net)
{
    if (l){
        for (int i = 0; i < net->batch; ++i){
            if (l->pool == MAX){
                Tensor *img = l->input[i];
                int height_col = (img->size[1] - l->ksize) / l->ksize + 1;
                int width_col = (img->size[0] - l->ksize) / l->ksize + 1;
                l->index = malloc((height_col*width_col*img->size[2])*sizeof(int));
                l->output[i] = forward_max_pool(l->input[i], l->ksize, l->index);
            } else {
                l->output[i] = forward_avg_pool(l->input[i], l->ksize);
            }
        }
    }
}

void backward_pooling_layer(Layer *l, Network *net)
{
    if (l){
        for (int i = 0; i < net->batch; ++i){
            Tensor *img = l->input[i];
            Tensor *delta = net->delta[i];
            if (l->pool == MAX){
                net->delta[i] = backward_max_pool(delta, l->ksize, img->size[1], img->size[0], l->index);
            } else {
                net->delta[i] = backward_avg_pool(delta, l->ksize, img->size[1], img->size[0]);
            }
            free_tensor(delta);
        }
    }
}

Layer make_pooling_layer(LayerParams *p, int batch, int h, int w, int c)
{
    Layer l = {0};
    l.type = POOLING;
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;
    Node *n = p->head;
    while (n){
        Params *param = n->val;
        if (0 == strcmp(param->key, "type")){
            if (0 == strcmp(param->val, "avg")) l.pool = AVG;
            else l.pool = MAX;
        } else if (0 == strcmp(param->key, "ksize")){
            l.ksize = atoi(param->val);
            l.stride = l.ksize;
        }
        n = n->next;
    }
    l.output_h = (l.input_h - l.ksize) / l.ksize + 1;
    l.output_w = (l.input_w - l.ksize) / l.ksize + 1;
    l.output_c = l.input_c;
    l.forward = forward_pooling_layer;
    l.backward = backward_pooling_layer;

    int size_o[] = {l.output_w, l.output_h, l.output_c};
    int size_d[] = {l.input_w, l.input_h, l.input_c};
    l.output = malloc(batch*sizeof(Tensor *));
    l.delta = malloc(batch*sizeof(Tensor *));
    for (int i = 0; i < batch; ++i){
        l.output[i] = tensor_x(3, size_o, 0);
        l.delta[i] = tensor_x(3, size_d, 0);
    }

    char *type;
    if (l.pool == AVG) type = "avg";
    else type = "max";
    fprintf(stderr, "  %s              %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", \
            type, l.ksize, l.ksize, l.ksize, l.input_h, l.input_w, l.input_c, \
            l.output_h, l.output_w, l.output_c);
    return l;
}