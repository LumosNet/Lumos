#include "data_log.h"

void logging_data(char *type, int h, int w, int c, void *data, FILE *buffer)
{
    if (0 == strcmp(type, "int")){
        logging_int_data(h, w, c, (int*)data, buffer);
    } else if (0 == strcmp(type, "float")){
        logging_float_data(h, w, c, (float*)data, buffer);
    }
}

void logging_int_data(int h, int w, int c, int *data, FILE *buffer)
{
    for (int k = 0; k < c; ++k){
        for (int i = 0; i < h; ++i){
            for (int j = 0; j < w; ++j){
                fprintf(buffer, "%d ", data[k*h*w+i*w+j]);
            }
            fprintf(buffer, "\n");
        }
        fprintf(buffer, "\n");
    }
}

void logging_float_data(int h, int w, int c, float *data, FILE *buffer)
{
    for (int k = 0; k < c; ++k){
        for (int i = 0; i < h; ++i){
            for (int j = 0; j < w; ++j){
                fprintf(buffer, "%f ", data[k*h*w+i*w+j]);
            }
            fprintf(buffer, "\n");
        }
        fprintf(buffer, "\n");
    }
}