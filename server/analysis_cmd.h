#ifndef ANALYSIS_CMD_H
#define ANALYSIS_CMD_H

#include <string.h>

#include "str_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ERROR 0
#define START 1
#define STOP 2
#define PAUSE 3
#define RUN 4
#define STATUS

int analysis_cmd(char *str);

#ifdef __cplusplus
}
#endif

#endif
