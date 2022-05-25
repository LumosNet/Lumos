#ifndef MACRO_H
#define MACRO_H

#ifdef  __cplusplus
extern "C" {
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define SGN(a) (sgn(a))

#define M_E		2.7182818284590452354

float sgn(float x);

#ifdef  __cplusplus
}
#endif

#endif