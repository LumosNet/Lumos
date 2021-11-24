#include "loss.h"

// label表示是第几类
int *one_hot_encoding(int n, int label)
{
    int *code = calloc(n, sizeof(int));
    code[label-1] = 1;
    return code;
}

float mse(Vector *yi, Vector *yh)
{
    Vector *y = subtract_ar(yi, yh);
    Vector *x = copy(y);
    transposition(x);
    Tensor *ts = gemm(x, y);
    float res = ts->data[0] / yi->num;
    del(y);
    del(x);
    del(ts);
    return res;
}

float mae(Vector *yi, Vector *yh)
{
    int sum = 0;
    for (int i = 0; i < yi->num; ++i){
        sum += fabs(yi->data[i] - yh->data[i]);
    }
    return sum / yi->num;
}

float huber(Vector *yi, Vector *yh, float theta)
{
    float huber = 0;
    for (int i = 0; i < yi->num; ++i){
        float differ = fabs(yi->data[i] - yh->data[i]);
        if (differ <= theta) huber += pow(differ, 2) / 2;
        else huber += theta * differ - 0.5 * pow(theta, 2);
    }
    return huber / yi->num;
}

float quantile(Vector *yi, Vector *yh, float r)
{
    float quant = 0;
    for (int i = 0; i < yi->num; ++i){
        float differ = fabs(yi->data[i] - yh->data[i]);
        if (yi->data[i] <  yh->data[i]){
            quant += (1-r) * differ;
        }
        else quant += r * differ;
    }
    return quant / yi->num;
}

float cross_entropy(Vector *yi, Vector *yh)
{
    float entropy = 0;
    for (int i = 0; i < yi->num; ++i){
        entropy += yi->data[i] * log(yh->data[i]);
    }
    return -entropy;
}

float hinge(Vector *yi, Vector *yh)
{
    float hinge = 0;
    for (int i = 0; i < yi->num; ++i){
        float x = 1 - SGN(yi->data[i])*yh->data[i];
        hinge += MAX(0, x);
    }
    return hinge;
}