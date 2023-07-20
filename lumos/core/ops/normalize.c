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

void gradient_normalize_mean(float *beta, float *variance, int num, float *mean_delta)
{
    for (int i = 0; i < num; ++i){
        mean_delta[i] = 1./sqrt(variance[i] + .00001f)*beta[i];
    }
}

void gradient_normalize_variance(float *beta, float *input, float *n_delta, float *mean, float *variance, int h, int w, int c, float *variance_delta)
{
    for (int i = 0; i < c; ++i){
        variance_delta[i] = 0;
        for (int j = 0; j < h*w; ++j){
            variance_delta[i] += n_delta[i*h*w + j]*(input[i*h*w + j]-mean[i]);
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.)) * beta[i];
    }
}

void gradient_normalize_cpu(float *input, float *mean, float *mean_delta, float *variance_delta, int h, int w, int c, float *n_delta, float *l_delta, float *space)
{
    for (int i = 0; i < c; ++i){
        space[i] = 0;
        for (int j = 0; j < h*w; ++j){
            l_delta[i*h*w + j] = n_delta[i*h*w + j] * mean_delta[i] + 2.0/(h*w)*(input[i*h*w + j]-mean[i])*variance_delta[i];
            space[i] += l_delta[i*h*w + j];
        }
    }
}

void gradient_normalize_layer(int h, int w, int c, float *l_delta, float *space)
{
    for (int i = 0; i < c; ++i){
        for (int j = 0; j < h*w; ++j){
            l_delta[i*h*w + j] -= 1/(h*w)*space[i];
        }
    }
}
