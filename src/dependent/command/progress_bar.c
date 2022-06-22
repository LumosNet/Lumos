#include "progress_bar.h"

void progress_bar(int n, int m)
{
    int p = (n/(float)m)*20;
    fprintf(stderr, "\r[");
    for (int i = 0; i < p; ++i){
        fprintf(stderr, "=");
    }
    for (int i = 0; i < 20-p; ++i){
        fprintf(stderr, " ");
    }
    fprintf(stderr, "]");
}