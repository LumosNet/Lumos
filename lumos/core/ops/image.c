#include "image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int *census_image_pixel(float *img, int w, int h, int c)
{
    int *num = calloc(256, sizeof(int));
    for (int i = 0; i < w * h * c; ++i)
    {
        num[(int)(img[i] * 255)] += 1;
    }
    return num;
}

int *census_channel_pixel(float *img, int w, int h, int c, int index_c)
{
    int *num = calloc(256, sizeof(int));
    int offset = (index_c - 1) * w * h;
    for (int i = 0; i < w * h; ++i)
    {
        num[(int)(img[i + offset] * 255)] += 1;
    }
    return num;
}

float *load_image_data(char *img_path, int *w, int *h, int *c)
{
    unsigned char *data = stbi_load(img_path, w, h, c, 0);
    if (c[0] > 3) c[0] = 3;
    float *im_new = malloc(w[0] * h[0] * c[0] * sizeof(float));
    for (int k = 0; k < c[0]; ++k)
    {
        for (int j = 0; j < h[0]; ++j)
        {
            for (int i = 0; i < w[0]; ++i)
            {
                int dst_index = i + w[0] * j + w[0] * h[0] * k;
                int src_index = k + c[0] * i + c[0] * w[0] * j;
                im_new[dst_index] = (float)data[src_index] / 255.;
            }
        }
    }
    free(data);
    return im_new;
}

void save_image_data(float *img, int w, int h, int c, char *savepath)
{
    int i, k;
    unsigned char *data = malloc((w * h * c) * sizeof(char));
    for (k = 0; k < c; ++k)
    {
        for (i = 0; i < w * h; ++i)
        {
            data[i * c + k] = (unsigned char)img[i + k * w * h];
        }
    }
    stbi_write_png(savepath, w, h, c, data, w * c);
    free(data);
}

void resize_im(float *img, int height, int width, int channel, int row, int col, float *space)
{
    if (height == row && width == col)
    {
        memcpy(space, img, height * width * channel * sizeof(float));
        return;
    }
    float *part = calloc(col * height * channel, sizeof(float));
    int r, c, k;
    float w_scale = (float)(width - 1) / (col - 1);
    float h_scale = (float)(height - 1) / (row - 1);
    for (k = 0; k < channel; ++k)
    {
        for (r = 0; r < height; ++r)
        {
            for (c = 0; c < col; ++c)
            {
                float val = 0;
                if (c == col - 1 || width == 1)
                {
                    val = img[(width * height) * k + r * width + width - 1];
                }
                else
                {
                    float sx = c * w_scale;
                    int ix = (int)sx;
                    float dx = sx - ix;
                    val = (1 - dx) * img[(width * height) * k + r * width + ix] + dx * img[(width * height) * k + r * width + ix + 1];
                }
                part[(col * height) * k + r * col + c] = val;
            }
        }
    }
    for (k = 0; k < channel; ++k)
    {
        for (r = 0; r < row; ++r)
        {
            float sy = r * h_scale;
            int iy = (int)sy;
            float dy = sy - iy;
            for (c = 0; c < col; ++c)
            {
                space[(row * col) * k + col * r + c] = (1 - dy) * part[(col * height) * k + col * iy + c];
            }
            if (r == row - 1 || height == 1)
                continue;
            for (c = 0; c < col; ++c)
            {
                if (c < col && r < row && k < channel)
                {
                    space[(col * row) * k + col * r + c] += dy * part[(col * height) * k + col * (iy + 1) + c];
                }
            }
        }
    }
    free(part);
}