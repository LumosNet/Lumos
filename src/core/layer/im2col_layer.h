#ifndef IM2COL_LAYER_H
#define IM2COL_LAYER_H

#ifdef __cplusplus
extern "C" {
#endif

Layer make_im2col_layer(CFGParams *p, int h, int w, int c);

void forward_im2col_layer(Layer l, float *workspace);
void backward_im2col_layer(Layer l, float *n_delta, float *workspace);

#ifdef __cplusplus
}
#endif

#endif