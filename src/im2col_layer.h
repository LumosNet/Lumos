#ifndef IM2COL_LAYER_H
#define IM2COL_LAYER_H

#include "lumos.h"
#include "parser.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer make_im2col_layer(LayerParams *p, int batch, int h, int w, int c);

#ifdef __cplusplus
}
#endif

#endif