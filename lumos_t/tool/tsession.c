#include "tsession.h"

SessionT *make_t_session(char *path)
{
    SessionT *sess = malloc(sizeof(SessionT));
    cJSON *CJbenchmark = get_benchmark(path);
    cJSON *CJpublic = get_public(CJbenchmark);
    
}