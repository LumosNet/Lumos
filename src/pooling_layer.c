#include "pooling_layer.h"

PoolingType load_pooling_type(char *pool)
{
    if (0 == strcmp(pool, "average")) return AVG;
    return MAX;
}