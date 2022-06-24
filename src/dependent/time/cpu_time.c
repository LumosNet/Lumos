#include "cpu_time.h"

double what_time_is_it_now()
{
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}