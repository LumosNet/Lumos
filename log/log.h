#ifndef LOG_H
#define LOG_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEBUG 1
#define INFO 2
#define WARNING 3
#define ERROR 4
#define CRITICAL 5

#define DEBUGCOLOR 8
#define INFOCOLOR 7
#define WARNINGCOLOR 6
#define ERRORCOLOR 4
#define CRITICALCOLOR 4

void logging(FILE *buf, int level, char *msg, int prt);

void logging_debug(FILE *buf, char *msg, int prt);
void logging_info(FILE *buf, char *msg, int prt);
void logging_warning(FILE *buf, char *msg, int prt);
void logging_error(FILE *buf, char *msg, int prt);
void logging_critical(FILE *buf, char *msg, int prt);

#ifdef __cplusplus
}
#endif
#endif
