#include "normalize.h"

void normalize_mean(float *data, int h, int w, int c, int subdivision, float *mean)
{
    for (int i = 0; i < c; ++i){
        mean[i] = 0;
        for (int j = 0; j < subdivision; ++j){
            for (int k = 0; k < h*w; ++k){
                mean[i] += data[j*h*w*c+i*h*w+k];
            }
        }
        mean[i] /= subdivision*h*w;
    }
}

void normalize_variance(float *data, int h, int w, int c, int subdivision, float *mean, float *variance)
{
    for (int i = 0; i < c; ++i){
        variance[i] = 0;
        for (int j = 0; j < subdivision; ++j){
            for (int k = 0; k < h*w; ++k){
                variance[i] += pow(data[j*h*w*c+i*h*w+k]-mean[i], 2);
            }
        }
        variance[i] /= subdivision*h*w;
    }
}

void normalize_cpu(float *data, float *mean, float *variance, int h, int w, int c, float *space)
{
    int offset = h*w;
    for (int i = 0; i < c; ++i){
        float *data_c = data + i*offset;
        float *space_c = space + i*offset;
        for (int j = 0; j < offset; ++j){
            space_c[j] = (data_c[j] - mean[i]) / (sqrt(variance[i] + .00001f));
        }
    }
}

void gradient_normalize_mean(float *n_delta, float *variance, int h, int w, int c, float *mean_delta)
{
    for (int i = 0; i < c; ++i){
        mean_delta[i] = 0;
        for (int j = 0; j < h*w; ++j){
            mean_delta[i] += n_delta[i*h*w+j];
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}

void gradient_normalize_variance(float *n_delta, float *input, float *mean, float *variance, int h, int w, int c, float *variance_delta)
{
    for (int i = 0; i < c; ++i){
        variance_delta[i] = 0;
        for (int j = 0; j < h*w; ++j){
            variance_delta[i] += n_delta[i*h*w+j]*(input[i*h*w+j]-mean[i]);
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}

void gradient_normalize_cpu(float *input, float *mean, float *variance, float *mean_delta, float *variance_delta, int h, int w, int c, float *n_delta, float *l_delta)
{
    for (int i = 0; i < c; ++i){
        for (int j = 0; j < h*w; ++j){
            l_delta[i*h*w+j] = n_delta[i*h*w+j] * 1./(sqrt(variance[i] + .00001f)) + variance_delta[i] * 2. * (input[i*h*w+j] - mean[i]) / (h*w) + mean_delta[i]/(h*w);
        }
    }
}

void update_scale(float *output, float *delta, int h, int w, int c, float rate, float *space)
{
    for (int i = 0; i < c; ++i){
        float sum = 0;
        for (int j = 0; j < h*w; ++j){
            sum += output[i*h*w+j]*delta[i*h*w+j];
        }
        space[i] += rate * sum;
    }
}

void update_bias(float *delta, int h, int w, int c, float rate, float *space)
{
    for (int i = 0; i < c; ++i){
        float sum = 0;
        for (int j = 0; j < h*w; ++j){
            sum += delta[i*h*w+j];
        }
        space[i] += rate * sum;
    }
}
