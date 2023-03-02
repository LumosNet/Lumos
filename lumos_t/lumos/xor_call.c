#include "xor_call.h"

void _xor_label2truth(char **label, float *truth)
{
    int x = atoi(label[0]);
    one_hot_encoding(1, x, truth);
}

void call_xor(void **params, void **ret)
{

}
