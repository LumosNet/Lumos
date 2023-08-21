#ifndef LABELS_H
#define LABELS_H

#include <stdio.h>
#include <stdlib.h>

#include "session.h"
#include "text_f.h"

#ifdef __cplusplus
extern "C" {
#endif

void bind_labelset(Session *sess, char *labelset_path_file);

#ifdef __cplusplus
}
#endif
#endif
