#include "shortcut.h"

void shortcut_cpu(float *add, int aw, int ah, int ac, float *out, int ow, int oh, int oc, float beta, float alpha, float *space)
{
    int istride = aw / ow;
    int ostride = ow / aw;
    if (istride < 1) istride = 1;
    if (ostride < 1) ostride = 1;
    int minw = aw < ow ? aw : ow;
    int minh = ah < oh ? ah : oh;
    int minc = ac < oc ? ac : oc;
    for (int i = 0; i < minc; ++i){
        for (int j = 0; j < minh; ++j){
            for (int k = 0; k < minw; ++k){
                int iindex = i*ah*aw + j*istride*aw + k*istride;
                int oindex = i*oh*ow + j*ostride*ow + k*ostride;
                space[oindex] = add[iindex]*alpha + out[oindex]*beta;
            }
        }
    }
}
