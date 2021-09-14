#include "gray_process.h"

#ifdef GRAY

Image *img_reversal(Image *img)
{
    Image *new_img = copy_image(img);
    for (int i = 0; i < img->num; ++i){
        new_img->data[i] = 1 - new_img->data[i];
    }
    return new_img;
}

Image *Log_transfer(Image *img, float c)
{
    Image *new_img = copy_image(img);
    for (int i = 0; i < img->num; ++i){
        new_img->data[i] = c * log(new_img->data[i]+1/255.);
    }
    return new_img;
}

Image *power_law(Image *img, float c, float gamma, float varepsilon)
{
    Image *new_img = copy_image(img);
    for (int i = 0; i < img->num; ++i){
        new_img->data[i] = c * pow((new_img->data[i] + varepsilon), gamma);
    }
    return new_img;
}

Image *contrast_stretch(Image *img, float r_max, float r_min, float s_max, float s_min)
{
    Image *new_img = copy_image(img);
    for (int i = 0; i < img->num; ++i){
        new_img->data[i] = (new_img->data[i] - r_min) / (r_max - r_min) * (s_max - s_min) + s_min;
    }
    return new_img;
}

Image *gray_slice_1(Image *img, float slice, float focus, float no_focus)
{
    Image *new_img = copy_image(img);
    for (int i = 0; i < img->num; ++i){
        if (new_img->data[i] >= slice) new_img->data[i] = focus;
        else new_img->data[i] = no_focus;
    }
    return new_img;
}

Image *gray_slice_2(Image *img, float slice, int flag, float focus)
{
    Image *new_img = copy_image(img);
    for (int i = 0; i < img->num; ++i){
        if (flag){
            if (new_img->data[i] >= slice) new_img->data[i] = focus;
        }
        else{
            if (new_img->data[i] <= slice) new_img->data[i] = focus;
        }
    }
    return new_img;
}

Image **bit8_slice(Image *img)
{
    Image **new_imgs = malloc(8*sizeof(Image*));
    for (int i = 0; i < 8; ++i){
        new_imgs[i] = create_image(img->size[0], img->size[1], img->size[2]);
    }
    for (int i = 0; i < img->num; ++i){
        binary *bits = int2binary((int)(img->data[i]*255));
        node *n = bits->tail;
        for (int j = 0; j < 8; ++j){
            if (n){
                new_imgs[j]->data[i] = n->data;
                n = n->prev;
            }
            else new_imgs[j]->data[i] = 0;
        }
    }
    return new_imgs;
}

Image *bit_restructure(Image **imgs, int *bit_index, int len)
{
    Image *img = create_image(imgs[0]->size[0], imgs[0]->size[1], imgs[0]->size[2]);
    for (int i = 0; i < len; ++i){
        Image *bit_img = imgs[i];
        for (int j = 0; j < img->num; ++j){
            img->data[j] += pow(2, bit_index[i]) * bit_img->data[j] / 255.;
        }
    }
    return img;
}

Image *histogram_equalization(Image *img)
{
    Image *new_img = create_image(img->size[0], img->size[1], img->size[2]);
    for (int i = 0; i < img->size[2]; ++i){
        __histogram_equalization_channel(img, new_img, i+1);
    }
    return new_img;
}

void __histogram_equalization_channel(Image *o_img, Image *a_img, int c)
{
    int p_num = o_img->size[0] * o_img->size[1];
    int *num = census_channel_pixel(o_img, c);
    int *gray_level = calloc(256, sizeof(int));
    for (int i = 0; i < 256; ++i){
        int n = 0;
        for (int j = 0; j <= i; ++j){
            n += num[j];
        }
        gray_level[i] = 255. / p_num * n;
    }
    int offset = (c - 1) * p_num;
    for (int i = 0; i < p_num; ++i){
        a_img->data[i+offset] = gray_level[(int)(o_img->data[i+offset]*255)] / 255.;
    }
    free(num);
    free(gray_level);
}

#endif

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
                    int index[] = {img->size[0]-1, r, k};
                    val = get_pixel(img, index);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    int index1[] = {ix, r, k};
                    int index2[] = {ix+1, r, k};
                    val = (1 - dx) * get_pixel(img, index1) + dx * get_pixel(img, index2);
                }
                int index[] = {c, r, k};
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
                int index[] = {c, iy, k};
                int index1[] = {c, r, k};
                float val = (1-dy) * get_pixel(part, index);
                change_pixel(resized, index1, val);
            }
            if(r == height-1 || img->size[1] == 1) continue;
            for(c = 0; c < width; ++c){
                int index[] = {c, iy+1, k};
                int index1[] = {c, r, k};
                float val = dy * get_pixel(part, index);
                int lindex = index_ts2ls(index1, img->dim, img->size);
                if (lindex >= 0) resized->data[lindex] += val;
            }
        }
    }
    del(part);
    return resized;
}