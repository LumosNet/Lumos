#include "vector.h"

float norm1_vt(Tensor *ts)
{
    float res = 0;
    for (int i = 0; i < ts->num; ++i){
        res += fabs(ts->data[i]);
    }
    return res;
}

float norm2_vt(Tensor *ts)
{
    float res = 0;
    for (int i = 0; i < ts->num; ++i){
        res += ts->data[i] * ts->data[i];
    }
    res = sqrt(res);
    return res;
}

float normp_vt(Tensor *ts, int p)
{
    float res = 0;
    for (int i = 0; i < ts->num; ++i){
        res += pow(ts->data[i], (double)p);
    }
    res = pow(res, (double)1/p);
    return res;
}

float infnorm_vt(Tensor *ts)
{
    float res = ts_max(ts);
    return res;
}

float ninfnorm_vt(Tensor *ts)
{
    float res = ts_min(ts);
    return res;
}