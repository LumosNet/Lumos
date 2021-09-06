#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "image.h"

Image *create_image(int w, int h, int c)
{
    int *size = malloc(3*sizeof(int));
    size[0] = w;
    size[1] = h;
    size[2] = c;
    Image *image = create_matrix(3, size);
    free(size);
    return image;
}

Image *copy_image(Image *img)
{
    return copy_matrix(img);
}

float min_pixel(Image *img)
{
    return get_matrix_min(img);
}

float max_pixel(Image *img)
{
    return get_matrix_max(img);
}

float pixel_mean(Image *img)
{
    return get_matrix_mean(img);
}

int pixel_num_image(Image *img, float x)
{
    return pixel_num_matrix(img, x);
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