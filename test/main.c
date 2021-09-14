#include "image.h"
#include "gray_process.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    printf("okoko\n");
    Image *img = load_image_data("D:/Lumos/lumos-matrix/test/data/01.jpeg");
    printf("%d %d %d\n", img->size[0], img->size[1], img->size[2]);
    Image *new = resize_im(img, 608, 608);
    printf("okoko\n");
    save_image_data(new, "D:/Lumos/lumos-matrix/test/data/03.png");
    printf("okoko\n");
    return 1;
}