#include "image.h"
#include "gray_process.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    Image *img = load_image_data("D:/Lumos/lumos-matrix/test/data/01.jpeg");
    Image *new = resize_im(img, 608, 608);
    save_image_data(new, "D:/Lumos/lumos-matrix/test/data/03.png");
    return 1;
}