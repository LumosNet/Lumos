#include "image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Tensor *create_image(int w, int h, int c)
{
    int *size = malloc(3*sizeof(int));
    size[0] = w;
    size[1] = h;
    size[2] = c;
    Tensor *image = tensor_x(3, size, 0);
    free(size);
    return image;
}

int *census_image_pixel(Tensor *img)
{
    int *num = calloc(256, sizeof(int));
    for (int i = 0; i < img->num; ++i){
        num[(int)(img->data[i]*255)] += 1;
    }
    return num;
}

int *census_channel_pixel(Tensor *img, int c)
{
    int *num = calloc(256, sizeof(int));
    int offset = (c - 1) * img->size[0] * img->size[1];
    for (int i = 0; i < img->size[0]*img->size[1]; ++i){
        num[(int)(img->data[i+offset]*255)] += 1;
    }
    return num;
}

int get_channels(Tensor *img)
{
    if (img->dim == 2) return 1;
    else if (img->dim) return img->size[2];
    return 0;
}

Tensor *load_image_data(char *img_path)
{
    int w, h, c;
    unsigned char *data = stbi_load(img_path, &w, &h, &c, 0);
    Tensor *im_new = create_image(w, h, c);
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

void save_image_data(Tensor *img, char *savepath)
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

Tensor *resize_ts_im(Tensor *img, int width, int height)
{
    Tensor *resize_tsd = create_image(width, height, img->size[2]);
    Tensor *part = create_image(width, img->size[1], img->size[2]);
    int r, c, k;
    float w_scale = (float)(img->size[0] - 1) / (width - 1);
    float h_scale = (float)(img->size[1] - 1) / (height - 1);
    for(k = 0; k < img->size[2]; ++k){
        for(r = 0; r < img->size[1]; ++r){
            for(c = 0; c < width; ++c){
                float val = 0;
                if(c == width-1 || img->size[0] == 1){
                    int index[] = {img->size[0], r+1, k+1};
                    val = ts_get_pixel(img, index);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    int index1[] = {ix+1, r+1, k+1};
                    int index2[] = {ix+2, r+1, k+1};
                    val = (1 - dx) * ts_get_pixel(img, index1) + dx * ts_get_pixel(img, index2);
                }
                int index[] = {c+1, r+1, k+1};
                ts_change_pixel(part, index, val);
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
                float val = (1-dy) * ts_get_pixel(part, index);
                ts_change_pixel(resize_tsd, index1, val);
            }
            if(r == height-1 || img->size[1] == 1) continue;
            for(c = 0; c < width; ++c){
                int index[] = {c+1, iy+2, k+1};
                int index1[] = {c+1, r+1, k+1};
                float val = dy * ts_get_pixel(part, index);
                int lindex = index_ts2ls(index1, resize_tsd->dim, resize_tsd->size);
                if (lindex) resize_tsd->data[lindex] += val;
            }
        }
    }
    free_tensor(part);
    return resize_tsd;
}