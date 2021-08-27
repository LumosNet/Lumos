#include "tensor.h"

tensor *copy(tensor *m)
{

}

int get_index(tensor *m, float x);
float get_pixel(tensor *m, int *index);
void resize(tensor *m, int dim, int *size);
void slice(tensor *m, float *workspace, int *dim_c, int **size_c);
void merge(tensor *m, tensor *n, int dim, int index, float *workspace);
float get_sum(tensor *m);
float get_min(tensor *m);
float get_max(tensor *m);
float get_mean(tensor *m);
void del(tensor *m);