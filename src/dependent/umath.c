#include "math.h"

float sgn(float x){
    if (x > 0) return 1;
    if (x == 0) return 0;
    if (x < 0) return -1;
    return 0;
}