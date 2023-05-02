#ifndef SHORTCUT_H
#define SHORTCUT_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C"{
#endif

void shortcut_cpu(float *add, int aw, int ah, int ac, float *out, int ow, int oh, int oc, float beta, float alpha, float *space);

#ifdef __cplusplus
}
#endif

#endif
