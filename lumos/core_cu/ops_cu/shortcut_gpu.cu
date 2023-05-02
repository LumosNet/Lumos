#include "shortcut_gpu.h"

__global__ void shortcut_kernel(float *add, int aw, int ah, int ac, float *out, int ow, int oh, int oc, float beta, float alpha, float *space, int minw, int minh, int minc)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= minw*minh*minc) return;
    int istride = aw / ow;
    int ostride = ow / aw;
    if (istride < 1) istride = 1;
    if (ostride < 1) ostride = 1;
    int i = index / (minh*minw);
    int j = index / minw;
    int k = index % minw;
    int iindex = i*ah*aw + j*istride*aw + k*istride;
    int oindex = i*oh*ow + j*ostride*ow + k*ostride;
    space[oindex] = add[iindex]*alpha + out[oindex]*beta;
}

void shortcut_gpu(float *add, int aw, int ah, int ac, float *out, int ow, int oh, int oc, float beta, float alpha, float *space)
{
    int minw = aw < ow ? aw : ow;
    int minh = ah < oh ? ah : oh;
    int minc = ac < oc ? ac : oc;
    shortcut_kernel<<<(minw*minh*minc + BLOCK - 1)/BLOCK, BLOCK>>>(add, aw, ah, ac, out, ow, oh, oc, beta, alpha, space, minw, minh, minc);
}
