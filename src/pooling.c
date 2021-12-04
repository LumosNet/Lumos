#include "pooling.h"

Image *forward_avg_pool(Image *img, int ksize)
{
    Tensor *channel = array_x(ksize, ksize, (float)1/(ksize*ksize));
    int height_col = (img->size[1] - ksize) / ksize + 1;
    int width_col = (img->size[0] - ksize) / ksize + 1;
    float *data = malloc((height_col*width_col*img->size[2])*sizeof(float));
    for (int i = 0; i < img->size[2]; ++i){
        int size[] = {img->size[0], img->size[1], 1};
        Image *x = tensor_x(3, size, 0);
        memcpy(x->data, img->data+img->size[0]*img->size[1]*i, img->size[0]*img->size[1]*sizeof(float));
        Image *pooling = convolutional(x, channel, 0, ksize);
        memcpy(data+height_col*width_col*i, pooling->data, pooling->num*sizeof(float));
        free_tensor(x);
        free_tensor(pooling);
    }
    int res_size[] = {width_col, height_col, img->size[2]};
    Image *res = tensor_list(3, res_size, data);
    free_tensor(channel);
    return res;
}

Image *forward_max_pool(Image *img, int ksize, int *index)
{
    Tensor *channel = array_x(ksize, ksize, (float)1/(ksize*ksize));
    int height_col = (img->size[1] - ksize) / ksize + 1;
    int width_col = (img->size[0] - ksize) / ksize + 1;
    float *data = malloc((height_col*width_col*img->size[2])*sizeof(float));
    for (int i = 0; i < img->size[2]; ++i){
        int size[] = {img->size[0], img->size[1], 1};
        Image *x = tensor_x(3, size, 0);
        memcpy(x->data, img->data+img->size[0]*img->size[1]*i, img->size[0]*img->size[1]*sizeof(float));
        Tensor *img2col = im2col(x, ksize, ksize, 0);
        for (int h = 0; h < img2col->size[1]; h++){
            float max = -999;
            int max_index = -1;
            for (int w = 0; w <img2col->size[0]; w++){
                int mindex = h*img2col->size[0] + w;
                if (img2col->data[mindex] > max){
                    max = img2col->data[mindex];
                    max_index = (i*img->size[1]*img->size[0])+(h/width_col*ksize+w/ksize)*img->size[0]+(h%width_col*ksize+w%ksize);
                }
            }
            data[height_col*width_col*i + h] = max;
            index[height_col*width_col*i + h] = max_index;
        }
        free_tensor(x);
        free_tensor(img2col);
    }
    int res_size[] = {width_col, height_col, img->size[2]};
    Image *res = tensor_list(3, res_size, data);
    free_tensor(channel);
    return res;
}

Image *backward_avg_pool(Image *img, int ksize, int height, int width)
{
    Image *origin = create_image(width, height, img->size[2]);
    for (int c = 0; c < img->size[2]; ++c){
        for (int i = 0; i < height; ++i){
            for (int j = 0; j < width; ++j){
                int height_index = i / ksize;
                int width_index = j / ksize;
                int index[] = {width_index+1, height_index+1, c+1};
                int indexl[] = {j+1, i+1, c+1};
                float val = (float)ts_get_pixel(img, index)/(ksize*ksize);
                ts_change_pixel(origin, indexl, val);
            }
        }
    }
    return origin;
}

Image *backward_max_pool(Image *img, int ksize, int height, int width, int *index)
{
    Image *origin = create_image(width, height, img->size[2]);
    for (int i = 0; i < img->num; ++i){
        origin->data[index[i]] = img->data[i];
    }
    return origin;
}