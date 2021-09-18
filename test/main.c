#include "image.h"
#include "gray_process.h"
#include "array.h"
#include "im2col.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    Image *img1 = load_image_data("D:/Lumos/lumos-matrix/test/data/05.jpg");
    // float list[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
    //                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    // int size[] = {4, 4, 2};
    // Image *img1 = tensor_list(3, size, list);
    int ksize = 2;

    int height_col = (img1->size[1] - ksize) / ksize + 1;
    int width_col = (img1->size[0] - ksize) / ksize + 1;
    int *index = malloc((height_col*width_col*img1->size[2])*sizeof(int));

    Image *new = forward_max_pool(img1, ksize, index);
    // tsprint(new);
    // save_image_data(new, "D:/Lumos/lumos-matrix/test/data/03.jpg");
    Image *origin = backward_max_pool(new, ksize, img1->size[1], img1->size[0], index);
    save_image_data(origin, "D:/Lumos/lumos-matrix/test/data/04.jpg");
    // tsprint(origin);
    return 1;
}