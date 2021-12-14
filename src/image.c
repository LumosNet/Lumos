#include "image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Tensor *create_image(int w, int h, int c)
{
    int size[] = {w, h, c};
    Tensor *image = tensor_x(3, size, 0);
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

void resize_im(float *img, int height, int width, int channel, int row, int col, float *space)
{
    float *part = calloc(col*height*channel, sizeof(float));
    int r, c, k;
    float w_scale = (float)(width - 1) / (col - 1);
    float h_scale = (float)(height - 1) / (row - 1);
    for(k = 0; k < channel; ++k){
        for(r = 0; r < height; ++r){
            for(c = 0; c < col; ++c){
                float val = 0;
                if(c == col-1 || width == 1){
                    val = img[(width*height)*k + r*width + width-1];
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * img[(width*height)*k + r*width + ix] + dx * img[(width*height)*k + r*width + ix + 1];
                }
                part[(col*height)*k + r*col + c] = val;
            }
        }
    }
    for(k = 0; k < channel; ++k){
        for(r = 0; r < row; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < col; ++c){
                space[(row*col)*k + col*r + c] = (1-dy) * part[(col*height)*k + col*iy + c];
            }
            if(r == row-1 || height == 1) continue;
            for(c = 0; c < col; ++c){
                if (c < col && r < row && k < channel){
                    space[(col*row)*k + col*r + c] += dy * part[(col*height)*k + col*(iy+1) + c];
                }
            }
        }
    }
    free(part);
}