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
                Image *img = l->input;
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
            Image *img = l->input;
            if (l->pool == MAX){
                l->input[i] = backward_max_pool(l->output[i], l->ksize, img->size[1], img->size[0], l->index);
            } else {
                l->input[i] = backward_avg_pool(l->output[i], l->ksize, img->size[1], img->size[0]);
            }
        }
    }
}