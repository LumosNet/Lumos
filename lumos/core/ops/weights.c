#include "weights.h"

void uniform_init(int seed, float a, float b, int num, float *space)
{
    uniform_list(a, b, seed, num, space);
}

void guass_init(int seed, float mean, float variance, int num, float *space)
{
    guass_list(mean, variance, seed, num, space);
}

void xavier_init(int seed, int inp, int out, float *space)
{
    guass_list(0, 1, seed, inp*out, space);
    multy_cpu(space, inp*out, sqrt(1/(float)inp), 1);
}

void kaiming_init(int seed, int inp, int out, float *space)
{
    guass_list(0, 1, seed, inp*out, space);
    multy_cpu(space, inp*out, sqrt(2/(float)inp), 1);
}
