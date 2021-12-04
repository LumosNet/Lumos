#include "bias.h"

void add_bias(Tensor *ts, Tensor *bias, int n, int size)
{
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < size; ++j){
            ts->data[i*size + j] += bias->data[i];
        }
    }
}