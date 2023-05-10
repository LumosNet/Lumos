#include "normalize.h"

void normalize_mean(float *data, int h, int w, int c, float *mean)
{
    int offset = h*w;
    for (int i = 0; i < c; ++i){
        float *data_c = data + i*offset;
        mean_cpu(data_c, offset, mean+i);
    }
}

void normalize_variance(float *data, int h, int w, int c, float *mean, float *variance)
{
    int offset = h*w;
    for (int i = 0; i < c; ++i){
        float *data_c = data + i*offset;
        variance_cpu(data_c, mean[i], offset, variance+i);
    }
}

void normalize_cpu(float *data, float *mean, float *variance, int h, int w, int c, float *space)
{
    int offset = h*w;
    for (int i = 0; i < c; ++i){
        float *data_c = data + i*offset;
        float *space_c = space + i*offset;
        for (int j = 0; j < offset; ++j){
            space_c[j] = (data_c[j] - mean[i]) / (sqrt(variance[i]) + .000001f);
        }
    }
}
