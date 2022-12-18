#ifndef IMAGE_CALL_H
#define IMAGE_CALL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image.h"

#ifdef  __cplusplus
extern "C" {
#endif

void call_census_image_pixel(void **params, void **ret);
void call_census_channel_pixel(void **params, void **ret);

void call_load_image_data(void **params, void **ret);
void call_save_image_data(void **params, void **ret);

void call_resize_im(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
