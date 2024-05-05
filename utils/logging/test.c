#include "data_log.h"

int main()
{
    int h = 4;
    int w = 5;
    int c = 10;
    float *data = malloc(h*w*c*sizeof(float));
    FILE *buffer = fopen("./utils/logging/test.txt", "w");
    for (int i = 0; i < h*w*c; ++i){
        data[i] = i*0.1+0.2;
    }
    logging_data("float", h, w, c, (void*)data, buffer);
}