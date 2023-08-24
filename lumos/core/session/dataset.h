#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <stdlib.h>

#include "session.h"
#include "image.h"
#include "label.h"
#include "text_f.h"

#ifdef __cplusplus
extern "C" {
#endif

void bind_dataset(Session *sess, char *dataset_path_file);
void bind_labelset(Session *sess, char *labelset_path_file);
int load_dataandlabel(Session *sess);

#ifdef __cplusplus
}
#endif
#endif
