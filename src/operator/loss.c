#include "loss.h"

float mse(Victor *yi, Victor *yh)
{
    Victor *y = subtract_ar(yi, yh);
    victor *x = copy(y);
    transposition(x);
    Tensor *ts = gemm(x, y);
    float res = ts->data[0] / yi->num;
    del(y);
    del(x);
    del(ts);
    return res;
}