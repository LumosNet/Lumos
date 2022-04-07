#ifndef DATA_H
#define DATA_H

#include "lumos.h"
#include "image.h"
#include "parser.h"

#ifdef __cplusplus
extern "C" {
#endif

void load_train_data(Network *net, int offset);
void load_train_path(Network *net, char *data_path, char *label_path);
void load_weights(Network *net, char *file);
void save_weights(Network *net, char *file);

#ifdef __cplusplus
}
#endif

#endif