#include "image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Image *create_image(int w, int h, int c)
{
    int *size = malloc(3*sizeof(int));
    size[0] = w;
    size[1] = h;
    size[2] = c;
    Image *image = tensor_x(3, size, 0);
    free(size);
    return image;
}

int *census_image_pixel(Image *img)
{
    int *num = calloc(256, sizeof(int));
    for (int i = 0; i < img->num; ++i){
        num[(int)(img->data[i]*255)] += 1;
    }
    return num;
}

int *census_channel_pixel(Image *img, int c)
{
    int *num = calloc(256, sizeof(int));
    int offset = (c - 1) * img->size[0] * img->size[1];
    for (int i = 0; i < img->size[0]*img->size[1]; ++i){
        num[(int)(img->data[i+offset]*255)] += 1;
    }
    return num;
}

Image *load_image_data(char *img_path)
{
    int w, h, c;
    unsigned char *data = stbi_load(img_path, &w, &h, &c, 0);
    Image *im_new = create_image(w, h, c);
    int i, j, k;
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im_new->data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    free(data);
    return im_new;
}

void save_image_data(Image *img, char *savepath)
{
    int i, k;
    unsigned char *data = malloc(img->num*sizeof(char));
    for(k = 0; k < img->size[2]; ++k){
        for(i = 0; i < img->size[0]*img->size[1]; ++i){
            data[i*img->size[2]+k] = (unsigned char) (255*img->data[i + k*img->size[0]*img->size[1]]);
        }
    }
    stbi_write_png(savepath, img->size[0], img->size[1], img->size[2], data, img->size[0] * img->size[2]);
    free(data);
}

Image *resize_im(Image *img, int width, int height)
{
    Image *resized = create_image(width, height, img->size[2]);
    Image *part = create_image(width, img->size[1], img->size[2]);
    int r, c, k;
    float w_scale = (float)(img->size[0] - 1) / (width - 1);
    float h_scale = (float)(img->size[1] - 1) / (height - 1);
    for(k = 0; k < img->size[2]; ++k){
        for(r = 0; r < img->size[1]; ++r){
            for(c = 0; c < width; ++c){
                float val = 0;
                if(c == width-1 || img->size[0] == 1){
                    int index[] = {img->size[0], r+1, k+1};
                    val = get_pixel(img, index);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    int index1[] = {ix+1, r+1, k+1};
                    int index2[] = {ix+2, r+1, k+1};
                    val = (1 - dx) * get_pixel(img, index1) + dx * get_pixel(img, index2);
                }
                int index[] = {c+1, r+1, k+1};
                change_pixel(part, index, val);
            }
        }
    }
    for(k = 0; k < img->size[2]; ++k){
        for(r = 0; r < height; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < width; ++c){
                int index[] = {c+1, iy+1, k+1};
                int index1[] = {c+1, r+1, k+1};
                float val = (1-dy) * get_pixel(part, index);
                change_pixel(resized, index1, val);
            }
            if(r == height-1 || img->size[1] == 1) continue;
            for(c = 0; c < width; ++c){
                int index[] = {c+1, iy+2, k+1};
                int index1[] = {c+1, r+1, k+1};
                float val = dy * get_pixel(part, index);
                int lindex = index_ts2ls(index1, resized->dim, resized->size);
                if (lindex) resized->data[lindex] += val;
            }
        }
    }
    del(part);
    return resized;
}

Image *forward_conv(Image *img, Array *channel, int pad, int stride)
{
    int height_col = (img->size[1] + 2*pad - channel->size[0]) / stride + 1;
    int width_col = (img->size[0] + 2*pad - channel->size[0]) / stride + 1;
    Array *img2col = im2col(img, channel->size[0], stride, pad);
    Array *k = copy(channel);
    resize_ar(k, channel->num, 1);
    Array *convolutional = gemm(img2col, k);
    int size[] = {width_col, height_col, 1};
    resize(convolutional, 3, size);
    del(k);
    return convolutional;
}

Image *forward_avg_pool(Image *img, int ksize)
{
    Array *channel = array_x(ksize, ksize, (float)1/(ksize*ksize));
    int height_col = (img->size[1] - ksize) / ksize + 1;
    int width_col = (img->size[0] - ksize) / ksize + 1;
    float *data = malloc((height_col*width_col*img->size[2])*sizeof(float));
    for (int i = 0; i < img->size[2]; ++i){
        int size[] = {img->size[0], img->size[1], 1};
        Image *x = tensor_x(3, size, 0);
        memcpy(x->data, img->data+img->size[0]*img->size[1]*i, img->size[0]*img->size[1]*sizeof(float));
        Image *pooling = forward_conv(x, channel, 0, ksize);
        memcpy(data+height_col*width_col*i, pooling->data, pooling->num*sizeof(float));
        del(x);
        del(pooling);
    }
    int res_size[] = {width_col, height_col, img->size[2]};
    Image *res = tensor_list(3, res_size, data);
    del(channel);
    return res;
}

Image *forward_max_pool(Image *img, int ksize, int *index)
{
    Array *channel = array_x(ksize, ksize, (float)1/(ksize*ksize));
    int height_col = (img->size[1] - ksize) / ksize + 1;
    int width_col = (img->size[0] - ksize) / ksize + 1;
    float *data = malloc((height_col*width_col*img->size[2])*sizeof(float));
    for (int i = 0; i < img->size[2]; ++i){
        int size[] = {img->size[0], img->size[1], 1};
        Image *x = tensor_x(3, size, 0);
        memcpy(x->data, img->data+img->size[0]*img->size[1]*i, img->size[0]*img->size[1]*sizeof(float));
        Array *img2col = im2col(x, ksize, ksize, 0);
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
        del(x);
        del(img2col);
    }
    int res_size[] = {width_col, height_col, img->size[2]};
    Image *res = tensor_list(3, res_size, data);
    del(channel);
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
                float val = (float)get_pixel(img, index)/(ksize*ksize);
                change_pixel(origin, indexl, val);
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