#include "analysis_cmd.h"

int analysis_cmd(char *str)
{
    if (0 == strcmp(str, "start")) return START;
    else if (0 == strcmp(str, "stop")) return STOP;
    else if (0 == strcmp(str, "pause")) return PAUSE;
    else if (0 == strcmp(str, "run")) return RUN;
    else if (0 == strcmp(str, "status")) return STATUS;
    else return ERROR;
}