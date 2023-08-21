#ifndef INPUT_H
#define INPUT_H

#include <stdio.h>
#include <stdlib.h>

#include "session.h"
#include "text_f.h"

#ifdef __cplusplus
extern "C" {
#endif

void bind_dataset(Session *sess, char *dataset_path_file);

#ifdef __cplusplus
}
#endif
#endif
