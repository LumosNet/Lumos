#include "image.h"
#include "gray_process.h"
#include "array.h"
#include "im2col.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    Image *img1 = load_image_data("D:/Lumos/lumos-matrix/test/data/05.jpg");
    // float list[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    // int size[] = {4, 4, 1};
    // Image *img = tensor_list(3, size, list);
    Image *new = avg_pool(img1, 10);
    save_image_data(new, "D:/Lumos/lumos-matrix/test/data/03.jpg");

    return 1;
}