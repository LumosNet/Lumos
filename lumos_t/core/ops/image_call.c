#include "image_call.h"

void call_census_image_pixel(void **params, void **ret)
{
    float *img = (float*)params[0];
    int *w = (int*)params[1];
    int *h = (int*)params[2];
    int *c = (int*)params[3];
    int *rvalue = census_image_pixel(img, w[0], h[0], c[0]);
    ret[0] = (void*)rvalue;
}

void call_census_channel_pixel(void **params, void **ret)
{
    float *img = (float*)params[0];
    int *w = (int*)params[1];
    int *h = (int*)params[2];
    int *c = (int*)params[3];
    int *index_c = (int*)params[4];
    int *rvalue = census_channel_pixel(img, w[0], h[0], c[0], index_c[0]);
    ret[0] = (void*)rvalue;
}

void call_load_image_data(void **params, void **ret)
{
    char *img_path = (char*)params[0];
    int *w = (int*)params[1];
    int *h = (int*)params[2];
    int *c = (int*)params[3];
    float *rvalue = load_image_data(img_path, w, h, c);
    ret[0] = (void*)w;
    ret[1] = (void*)h;
    ret[2] = (void*)c;
    ret[3] = (void*)rvalue;
}

void call_save_image_data(void **params, void **ret)
{
    float *img = (float*)params[0];
    int *w = (int*)params[1];
    int *h = (int*)params[2];
    int *c = (int*)params[3];
    char *savepath = (char*)params[4];
    save_image_data(img, w[0], h[0], c[0], savepath);
    float *rvalue = load_image_data(savepath, w, h, c);
    ret[0] = (void*)w;
    ret[1] = (void*)h;
    ret[2] = (void*)c;
    ret[3] = (void*)rvalue;
}

void call_resize_im(void **params, void **ret)
{
    float *img = (float*)params[0];
    int *height = (int*)params[1];
    int *width = (int*)params[2];
    int *channel = (int*)params[3];
    int *row = (int*)params[4];
    int *col = (int*)params[5];
    float *space = (float*)params[6];
    resize_im(img, height[0], width[0], channel[0], row[0], col[0], space);
    ret[0] = (void*)space;
}
