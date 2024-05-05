#include "progress_bar.h"

void progress_bar(int n, int m, double time, float loss)
{
    int p = (n / (float)m) * 20;
    fprintf(stderr, "\r[");
    for (int i = 0; i < p; ++i)
    {
        fprintf(stderr, "=");
    }
    for (int i = 0; i < 20 - p; ++i)
    {
        fprintf(stderr, " ");
    }
    fprintf(stderr, "]");
    fprintf(stderr, "  Time: %.3lfs", time);
    fprintf(stderr, "  Loss: %.3f", loss);
}