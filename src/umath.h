#ifndef MACRO_H
#define MACRO_H

#ifdef  __cplusplus
extern "C" {
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define SGN(a) (sgn(a))

float sgn(float x);

#ifdef  __cplusplus
}
#endif

#endif