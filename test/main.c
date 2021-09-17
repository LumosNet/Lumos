#include "image.h"
#include "gray_process.h"
#include "array.h"
#include "im2col.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    // Image *img1 = load_image_data("D:/Lumos/lumos-matrix/test/data/05.jpg");
    // float *data = malloc((img1->size[0]*img1->size[1])*sizeof(float));
    // memcpy(data, img1->data, (img1->size[0]*img1->size[1])*sizeof(float));

    // int size_img[] = {img1->size[0], img1->size[1], 1};
    // Image *img = tensor_list(3, size_img, data);
    // Array *imtocol = im2col(img, 3, 1, 0);
    // printf("im2col\n");
    // float list_channel[] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
    // Array *channel = array_list(9, 1, list_channel);

    // int height_col = (img->size[1] + 2*0 - 3) / 1 + 1;
    // int width_col = (img->size[0] + 2*0 - 3) / 1 + 1;

    // Image *new = gemm(imtocol, channel);
    // printf("gemm\n");
    // int size[] = {width_col, height_col, 1};
    // resize(new, 3, size);
    // printf("resize\n");
    // // save_image_data(new, "D:/Lumos/lumos-matrix/test/data/03.jpg");

    // FILE *fp;
    // fp = fopen("D:/Lumos/lumos-matrix/test/data/test.txt", "wt+");
    // for(int i=0; i<new->num; i++){
    //     fprintf(fp,"%f\n", new->data[i]);
    // }

    float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, \
                    17, 18, 19, 20, 21, 22, 23 ,24, 25, 26, 27, 28, 29, 30, 31, 32};
    float channel_data[] = {1, 0, -1, 1, 0, -1, 1, 0, -1};

    int size_img[] = {4, 4, 2};
    Image *img = tensor_list(3, size_img, data);

    int pad = 0;

    int height_col = (img->size[1] + 2*pad - 3) / 1 + 1;
    int width_col = (img->size[0] + 2*pad - 3) / 1 + 1;

    Array *channel = array_list(9, 1, channel_data);
    Array *imgtocol = im2col(img, 3, 1, pad);

    Array *new = gemm(imgtocol, channel);
    int size[] = {width_col, height_col, 1};
    resize(new, 3, size);
    tsprint(new);

    return 1;
}