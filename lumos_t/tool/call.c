#include "call.h"

void call_ops(char *interface, void **params, void **ret)
{
    if (0 == strcmp(interface, "add_bias")){
        call_add_bias(params, ret);
    }
}

#ifdef GPU
void call_cu_ops(char *interface, void **params, void **ret)
{
    if (0 == strcmp(interface, "add_bias")){
        call_add_bias_gpu(params, ret);
    }
}
#endif
